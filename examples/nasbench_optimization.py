import argparse
import os
import warnings

import torch

from comprehensive_nas.bo.acquisition_function_optimization.random_sampler import (
    RandomSampler as Sampler,
)
from comprehensive_nas.bo.acqusition_functions import AcquisitionMapping
from comprehensive_nas.bo.benchmarks import BenchmarkMapping
from comprehensive_nas.bo.benchmarks.nas.nb_configfiles.api.nas_201_api import (
    NASBench201API as API201,
)
from comprehensive_nas.bo.benchmarks.nas.nb_configfiles.api.nas_301_api import (
    NASBench301API as API301,
)
from comprehensive_nas.bo.kernels import GraphKernelMapping, StationaryKernelMapping
from comprehensive_nas.bo.models.gp import ComprehensiveGP
from comprehensive_nas.bo.optimizer import BayesianOptimization
from comprehensive_nas.rs.optimizer import RandomSearch
from comprehensive_nas.utils.util import StatisticsTracker

warnings.simplefilter("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="CNAS")
parser.add_argument(
    "--dataset",
    default="nasbench301",
    help="The benchmark dataset to run the experiments. "
    'options = ["nasbench201", "nasbench301", "hartmann3", '
    '"hartmann6", "counting_ones"].',
    choices=BenchmarkMapping.keys(),
)
parser.add_argument(
    "--task",
    default="None",
    type=str,
    choices=["None", "cifar10-valid", "cifar100", "ImageNet16-120"],
    help="the benchmark task *for nasbench201 only*.",
)
parser.add_argument(
    "--n_repeat", type=int, default=1, help="number of repeats of experiments"
)
parser.add_argument(
    "--n_init", type=int, default=40, help="number of initialising points"
)
parser.add_argument(
    "--max_iters", type=int, default=100, help="number of iterations for the search"
)
parser.add_argument(
    "-ps",
    "--pool_size",
    type=int,
    default=100,
    help="number of candidates generated at each iteration",
)
parser.add_argument(
    "--mutate_size",
    type=int,
    help="number of mutation candidates. By default, half of the pool_size "
    "is generated from mutation.",
)
parser.add_argument(
    "--pool_strategy",
    default="random",
    help="the pool generation strategy. Options: random," "mutate",
)
parser.add_argument(
    "-s",
    "--strategy",
    default="gbo",
    help="optimisation strategy: option: gbo (graph bo), " "random (random search)",
)
parser.add_argument(
    "-a",
    "--acquisition",
    default="UCB",
    help="the acquisition function for the BO algorithm. option: " "UCB, EI, AEI",
)
parser.add_argument(
    "-kg",
    "--graph_kernels",
    default=["wl"],
    nargs="+",
    help="graph kernel to use. This can take multiple input arguments, and "
    "the weights between the kernels will be automatically determined"
    " during optimisation (weights will be deemed as additional "
    "hyper-parameters.",
)
parser.add_argument(
    "-oa", "--optimal_assigment", action="store_true", help="Whether to optimize arch"
)
parser.add_argument(
    "-kh", "--hp_kernel", default="m52", help="hp kernel to use. Can be [rbf, m52, m32]"
)
parser.add_argument(
    "-dsk",
    "--domain_se_kernel",
    default="m52",
    help="Successive Embedding kernel on the domain to use. Can be [rbf, m52, m32]",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Number of samples to evaluate"
)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--optimize_arch", action="store_true", help="Whether to optimize arch"
)
parser.add_argument("--optimize_hps", action="store_true", help="Whether to optimize hps")
parser.add_argument("--cuda", action="store_true", help="Whether to use GPU acceleration")
parser.add_argument(
    "--no_isomorphism",
    action="store_true",
    help="Whether to allow mutation to return" "isomorphic architectures",
)
parser.add_argument(
    "--maximum_noise",
    default=0.01,
    type=float,
    help="The maximum amount of GP jitter noise variance",
)
parser.add_argument(
    "--log", action="store_true", help="Whether to report the results in log scale"
)

parser.add_argument(
    "--save_path",
    default=os.path.dirname(os.path.realpath(__file__)) + "/results/",
    help="path to save log file",
)
parser.add_argument("--plot", action="store_true", help="Whether to plot the procedure")
parser.add_argument(
    "--api_data_path",
    default="data/NAS-Bench-201-v1_0-e61699.pth",
    help="Full path to data file. Only needed for tabular/surrogate benchmarks!",
)
parser.add_argument(
    "--fixed_query_seed",
    type=int,
    default=None,
    help="Whether to use deterministic objective function as NAS-Bench-101 has 3 different seeds for "
    "validation and test accuracies. Options in [None, 0, 1, 2]. If None the query will be "
    "random.",
)


def run_experiment(args):
    if args.cuda and torch.cuda.is_available():
        device = "cuda"  # pylint: disable=unused-variable
    else:
        device = "cpu"  # pylint: disable=unused-variable

    # Initialise the objective function and its optimizer.
    api = None
    if args.dataset == "nasbench201":
        api = API201(args.data_path, verbose=args.verbose)
    elif args.dataset == "nasbench301":
        api = API301()
    objective = BenchmarkMapping[args.dataset](
        log_scale=args.log,
        seed=args.seed,
        optimize_arch=args.optimize_arch,
        optimize_hps=args.optimize_hps,
    )
    assert args.pool_strategy in [
        "random",
        "mutate",
    ]
    initial_design = Sampler(objective)
    acquisition_function_opt = Sampler(objective)

    # Initialise the optimizer strategy.
    assert args.strategy in ["random", "gbo"]
    if args.strategy == "random":
        optimizer = RandomSearch(acquisition_function_opt=acquisition_function_opt)
    elif args.strategy == "gbo":
        kern = []
        if args.optimize_arch:
            for kg in args.graph_kernels:
                kern.append(
                    GraphKernelMapping[kg](
                        oa=args.optimal_assigment,
                        se_kernel=StationaryKernelMapping[args.domain_se_kernel],
                    )
                )
        hp_kern = None
        if args.optimize_hps:
            hp_kern = []
            if objective.has_continuous_hp:
                hp_kern.append(StationaryKernelMapping[args.hp_kernel])
            if objective.has_categorical_hp:
                hp_kern.append(StationaryKernelMapping["hm"])

        surrogate_model = ComprehensiveGP(
            graph_kernels=kern, hp_kernels=hp_kern, verbose=args.verbose
        )
        acquisition_function = AcquisitionMapping[args.acquisition](
            surrogate_model=surrogate_model, strategy=args.strategy
        )
        optimizer = BayesianOptimization(
            surrogate_model=surrogate_model,
            acquisition_function=acquisition_function,
            acquisition_function_opt=acquisition_function_opt,
        )
    else:
        raise Exception(f"Optimizer {args.strategy} is not yet implemented!")

    experiments = StatisticsTracker(args)

    for seed in range(args.seed, args.seed + args.n_repeat):
        experiments.reset(seed)

        # Take n_init random samples
        x_configs = initial_design.sample(pool_size=args.n_init)

        # & evaluate
        y_np_list = [config_.query(dataset_api=api) for config_ in x_configs]
        # y = torch.Tensor([y[0] for y in y_np_list]).float()
        y = [y[0] for y in y_np_list]
        train_details = [y[1] for y in y_np_list]

        # Initialise the GP surrogate
        optimizer.initialize_model(x_configs=x_configs, y=y)

        # Main optimization loop
        while experiments.has_budget():

            # Propose new location to evaluate
            next_x, opt_details = optimizer.propose_new_location(
                args.batch_size, args.pool_size
            )
            pool = opt_details["pool"]

            # Evaluate this location from the objective function
            detail = [config_.query(dataset_api=api) for config_ in next_x]
            next_y = [y[0] for y in detail]
            train_details += [y[1] for y in detail]

            if optimizer.surrogate_model is not None:
                pool.extend(next_x)

            x_configs.extend(next_x)
            # y = torch.cat((y, torch.tensor(next_y).view(-1))).float()
            y.extend(next_y)

            # Update the GP Surrogate
            optimizer.update_model(x_configs=x_configs, y=y)
            experiments.print(x_configs, y, next_y, train_details)

            experiments.next_iteration()

        experiments.save_results()

        if args.plot:
            pass


if __name__ == "__main__":
    args = parser.parse_args()
    options = vars(args)
    print(options)

    assert args.dataset in BenchmarkMapping.keys(), "Required dataset is not implemented!"

    run_experiment(args)
