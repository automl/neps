import argparse
import warnings
from copy import deepcopy

import torch

from comprehensive_nas.bo.acquisition_function_optimization.sampler import Sampler
from comprehensive_nas.bo.acqusition_functions import AcquisitionMapping
from comprehensive_nas.bo.benchmarks import *
from comprehensive_nas.bo.kernel_operators import (
    GraphKernelMapping,
    StationaryKernelMapping,
)
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
parser.add_argument("--save_path", default="demo/results/", help="path to save log file")
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
parser.add_argument(
    "--n_graph_kernels", type=int, default=1, help="How many graph kernels to use"
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
args = parser.parse_args()


if args.cuda and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Initialise the objective function and its optimizer.
assert args.dataset in BenchmarkMapping.keys(), "Required dataset is not implemented!"
objective = BenchmarkMapping[args.dataset](log_scale=args.log, seed=args.seed)
assert args.pool_strategy in [
    "random",
    "mutate",
]
acquisition_function_opt = Sampler(args, objective)

# Initialise the optimizer strategy.
assert args.strategy in ["random", "gbo"]
if args.strategy == "random":
    optimizer = RandomSearch(acquisition_function_opt=acquisition_function_opt)
elif args.strategy == "gbo":
    kern = []
    if args.optimize_arch:
        for _ in range(args.n_graph_kernels):
            kern.append(
                GraphKernelMapping["wl"](
                    se_kernel=StationaryKernelMapping[args.domain_se_kernel]
                )
            )
    hp_kernel = StationaryKernelMapping[args.hp_kernel] if args.optimize_hps else None

    surrogate_model = ComprehensiveGP(
        graph_kernels=kern, hp_kernel=hp_kernel, verbose=args.verbose
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
    optimizer = None

experiments = StatisticsTracker(args)

for seed in range(args.seed, args.seed + args.n_repeat):
    experiments.reset(seed)

    # Take n_init random samples
    # TODO acquisiton function opt can be different to intial design!
    x_configs = acquisition_function_opt.sample(pool_size=args.n_init)

    # & evaluate
    y_np_list = [objective.eval(config_) for config_ in x_configs]
    y = torch.tensor([y[0] for y in y_np_list]).float()
    train_details = [y[1] for y in y_np_list]

    # Initialise the GP surrogate
    optimizer.initialize_model(
        x_configs=deepcopy(x_configs),
        y=deepcopy(y),
        optimize_arch=args.optimize_arch,
        optimize_hps=args.optimize_hps,
    )

    # Main optimization loop
    while experiments.has_budget():

        # Propose new location to evaluate
        next_x, pool = optimizer.propose_new_location(args.batch_size, args.pool_size)

        # Evaluate this location from the objective function
        detail = [objective.eval(config_) for config_ in next_x]
        next_y = [y[0] for y in detail]
        train_details += [y[1] for y in detail]

        if optimizer.surrogate_model is not None:
            pool.extend(next_x)

        x_configs.extend(next_x)
        y = torch.cat((y, torch.tensor(next_y).view(-1))).float()

        # Update the GP Surrogate
        optimizer.update_model(x_configs=deepcopy(x_configs), y=deepcopy(y))
        experiments.print(x_configs, y, next_y, train_details)

        experiments.next_iteration()

    experiments.save_results()
