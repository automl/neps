import argparse
import os
import warnings

try:
    import torch
except ModuleNotFoundError:
    from comprehensive_nas.utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)


from comprehensive_nas.optimizers.bayesian_optimization.acquisition_function_optimization import (
    AcquisitionOptimizerMapping,
)
from comprehensive_nas.optimizers.bayesian_optimization.acqusition_functions import (
    AcquisitionMapping,
)
from comprehensive_nas.optimizers.bayesian_optimization.benchmarks import (
    BenchmarkMapping,
    PipelineFunctionMapping,
)
from comprehensive_nas.optimizers.bayesian_optimization.benchmarks.nas.nb_configfiles.api import (
    APIMapping,
)
from comprehensive_nas.optimizers.bayesian_optimization.kernels import (
    GraphKernelMapping,
    StationaryKernelMapping,
)
from comprehensive_nas.optimizers.bayesian_optimization.models.gp import ComprehensiveGP
from comprehensive_nas.optimizers.bayesian_optimization.optimizer import (
    BayesianOptimization,
)
from comprehensive_nas.optimizers.random_search.optimizer import RandomSearch
from comprehensive_nas.utils.util import Experimentator, StatisticsTracker

from comprehensive_nas.search_spaces.search_space import SearchSpace
from comprehensive_nas.search_spaces import HyperparameterMapping

warnings.simplefilter("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="CNAS")
parser.add_argument(
    "--dataset",
    default="nasbench201",
    help="The benchmark dataset to run the experiments.",
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
    "--n_repeat", type=int, default=20, help="number of repeats of experiments"
)
parser.add_argument(
    "--n_init", type=int, default=30, help="number of initialising points"
)
parser.add_argument(
    "--max_iters", type=int, default=17, help="number of iterations for the search"
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
    default="mutation",
    help="the pool generation strategy. Options: random," "mutation, evolution",
    choices=AcquisitionOptimizerMapping.keys(),
)
parser.add_argument(
    "-s",
    "--strategy",
    default="gbo",
    help="optimisation strategy: option: gbo (graph bo), " "random (random search)",
    choices=["random", "gbo"],
)
parser.add_argument(
    "-a",
    "--acquisition",
    default="EI",
    help="the acquisition function for the BO algorithm.",
    choices=AcquisitionMapping.keys(),
)
parser.add_argument(
    "-kg",
    "--graph_kernels",
    default=[],
    nargs="+",
    help="graph kernel to use. This can take multiple input arguments, and "
    "the weights between the kernels will be automatically determined"
    " during optimisation (weights will be deemed as additional "
    "hyper-parameters.",
    choices=GraphKernelMapping.keys(),
)
parser.add_argument(
    "-oa", "--optimal_assignment", action="store_true", help="Whether to optimize arch"
)
parser.add_argument(
    "-kh",
    "--hp_kernels",
    default=[],
    nargs="+",
    help="hp kernel to use.",
    choices=StationaryKernelMapping.keys(),
)
parser.add_argument(
    "-dsk",
    "--domain_se_kernel",
    default=None,
    help="Successive Embedding kernel on the domain to use.",
    choices=[None] + list(StationaryKernelMapping.keys()),
)
parser.add_argument(
    "--batch_size", type=int, default=5, help="Number of samples to evaluate"
)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--seed", type=int, default=5)
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
    default="../comprehensive_nas/optimizers/bayesian_optimization/"
    "benchmarks/nas/nb_configfiles/data/NAS-Bench-201-v1_0-e61699.pth",
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
        assert args.dataset in APIMapping.keys()
        api = APIMapping[args.dataset](args.api_data_path, verbose=args.verbose)
    elif args.dataset == "nasbench301":
        assert args.dataset in APIMapping.keys()
        api = APIMapping[args.dataset]

    # branin2
    objective = SearchSpace(
        HyperparameterMapping['float'](name='x1', lower=-5, upper=10, log=False),
        HyperparameterMapping['float'](name='x2', lower=0, upper=15, log=False),
    )

    # hartmann3
    objective = SearchSpace(
        HyperparameterMapping['float'](name='x1', lower=0, upper=1, log=False),
        HyperparameterMapping['float'](name='x2', lower=0, upper=1, log=False),
        HyperparameterMapping['float'](name='x3', lower=0, upper=1, log=False),
    )
    #
    # hartmann6
    objective = SearchSpace(
        HyperparameterMapping['float'](name='x1', lower=0, upper=1, log=False),
        HyperparameterMapping['float'](name='x2', lower=0, upper=1, log=False),
        HyperparameterMapping['float'](name='x3', lower=0, upper=1, log=False),
        HyperparameterMapping['float'](name='x4', lower=0, upper=1, log=False),
        HyperparameterMapping['float'](name='x5', lower=0, upper=1, log=False),
        HyperparameterMapping['float'](name='x6', lower=0, upper=1, log=False),
    )
    #
    # counting_ones
    objective = SearchSpace(
        HyperparameterMapping['float'](name='x1', lower=0, upper=1, log=False),
        HyperparameterMapping['float'](name='x2', lower=0, upper=1, log=False),
        HyperparameterMapping['float'](name='x3', lower=0, upper=1, log=False),
        # HyperparameterMapping['float'](name='x4', lower=0, upper=1, log=False),
        # HyperparameterMapping['float'](name='x5', lower=0, upper=1, log=False),
        HyperparameterMapping['categorical'](name='x6', choices=[0, 1]),
        HyperparameterMapping['categorical'](name='x7', choices=[0, 1]),
        HyperparameterMapping['categorical'](name='x8', choices=[0, 1]),
        # HyperparameterMapping['categorical'](name='x9', choices=[0, 1]),
        # HyperparameterMapping['categorical'](name='x10', choices=[0, 1]),
        HyperparameterMapping['integer'](name='x11', lower=0, upper=1, log=False),
        HyperparameterMapping['integer'](name='x12', lower=0, upper=1, log=False),
        HyperparameterMapping['integer'](name='x13', lower=0, upper=1, log=False),
        # HyperparameterMapping['integer'](name='x14', lower=0, upper=1, log=False),
        # HyperparameterMapping['integer'](name='x15', lower=0, upper=1, log=False),
        HyperparameterMapping['constant'](name='x16', value=1),
        HyperparameterMapping['constant'](name='x17', value=1),
        HyperparameterMapping['constant'](name='x18', value=1),
        HyperparameterMapping['constant'](name='x19', value=1),
        HyperparameterMapping['constant'](name='x20', value=1),
    )

    initial_design = AcquisitionOptimizerMapping["random"](objective)

    if args.dataset in PipelineFunctionMapping:
        run_pipeline_fn = PipelineFunctionMapping[args.dataset]()
    else:
        run_pipeline_fn = None

    # Initialise the optimizer strategy.
    if args.strategy == "random":
        optimizer = RandomSearch(
            acquisition_function_opt=AcquisitionOptimizerMapping["random"](objective)
        )
    elif args.strategy == "gbo":
        kern = [
            GraphKernelMapping[kg](
                oa=args.optimal_assignment,
                se_kernel=None
                if args.domain_se_kernel is None
                else StationaryKernelMapping[args.domain_se_kernel],
            )
            for kg in args.graph_kernels
        ]
        hp_kern = [StationaryKernelMapping[kh]() for kh in args.hp_kernels]

        surrogate_model = ComprehensiveGP(
            graph_kernels=kern, hp_kernels=hp_kern, verbose=args.verbose
        )
        acquisition_function = AcquisitionMapping[args.acquisition](
            surrogate_model=surrogate_model
        )

        acquisition_function_opt = None
        if args.pool_strategy == "mutation":
            acquisition_function_opt = AcquisitionOptimizerMapping[args.pool_strategy](
                objective,
                acquisition_function,
                n_mutate=args.mutate_size,
                allow_isomorphism=args.no_isomorphism,
            )
        elif args.pool_strategy == "evolution":
            acquisition_function_opt = AcquisitionOptimizerMapping[args.pool_strategy](
                objective,
                acquisition_function,
            )
        elif args.pool_strategy == "random":
            acquisition_function_opt = AcquisitionOptimizerMapping[args.pool_strategy](
                objective,
                acquisition_function,
            )
        optimizer = BayesianOptimization(
            surrogate_model=surrogate_model,
            acquisition_function_opt=acquisition_function_opt,
            return_opt_details=False,
        )
    elif args.strategy == "re":
        optimizer = RegularizedEvolution(objective)
    else:
        raise Exception(f"Optimizer {args.strategy} is not yet implemented!")

    experimenter = Experimentator(args.max_iters, args.seed)
    tracker = StatisticsTracker(
        args, args.save_path, args.log, is_gbo=args.strategy == "gbo"
    )

    for seed in range(args.n_repeat):
        tracker = evaluate_seed(
            api=api,
            args=args,
            experimenter=experimenter,
            initial_design=initial_design,
            optimizer=optimizer,
            seed=seed,
            tracker=tracker,
            run_pipeline_fn=run_pipeline_fn,
        )
        # tracker.save_results()


def evaluate_seed(
    api,
    args,
    experimenter,
    initial_design,
    optimizer,
    seed,
    tracker,
    run_pipeline_fn=None,
):
    experimenter.reset(seed)
    tracker.reset()
    # Take n_init random samples
    x_configs = initial_design.sample(pool_size=args.n_init)

    # & evaluate
    y_np_list = [
        run_pipeline_fn(config_)
        if run_pipeline_fn is not None
        else config_.query(dataset_api=api)
        for config_ in x_configs
    ]

    # tracker.save_checkpoint(checkpoint_data=y_np_list)

    y = [_y["loss"] for _y in y_np_list]
    train_details = [_y["info_dict"] for _y in y_np_list]
    y_val = [_y["val_score"] for _y in train_details]
    y_test = [_y["test_score"] for _y in train_details]

    tracker.update(
        x=x_configs,
        y_eval=y_val,
        y_eval_cur=y_val,
        y_test=y_test,
        y_test_cur=y_test,
        train_details=train_details,
        opt_details=None,
    )
    tracker.print(False, True)

    # Initialise the GP surrogate
    optimizer.initialize_model(x_configs=x_configs, y=y)

    # Main optimization loop
    while experimenter.has_budget():

        # Propose new location to evaluate
        next_x, opt_details = optimizer.propose_new_location(
            **{"batch_size": args.batch_size, "pool_size": args.pool_size}
        )

        # Evaluate this location from the objective function
        detail = [
            run_pipeline_fn(config_)
            if run_pipeline_fn is not None
            else config_.query(dataset_api=api)
            for config_ in next_x
        ]

        # tracker.save_checkpoint(checkpoint_data=detail)

        next_y = [_y["loss"] for _y in detail]
        y_info_dict = [_y["info_dict"] for _y in detail]
        next_val = [_y["val_score"] for _y in y_info_dict]
        next_test = [_y["test_score"] for _y in y_info_dict]
        train_details += y_info_dict

        if opt_details is not None:
            pool = opt_details["pool"]

            pool_vals = [
                run_pipeline_fn(config_)
                if run_pipeline_fn is not None
                else config_.query(dataset_api=api)
                for config_ in pool
            ]

            opt_details["pool_vals"] = pool_vals
            pool.extend(next_x)

        x_configs.extend(next_x)
        y.extend(next_y)
        y_val.extend(next_val)
        y_test.extend(next_test)

        # Update the GP Surrogate
        optimizer.update_model(x_configs=x_configs, y=y)

        tracker.update(
            x=x_configs,
            y_eval=y_val,
            y_eval_cur=next_y,
            y_test=y_test,
            y_test_cur=next_test,
            train_details=train_details,
            opt_details=opt_details,
        )
        tracker.print(args.plot if args.strategy == "gbo" else False, True)

        experimenter.next_iteration()
    return tracker


if __name__ == "__main__":
    args = parser.parse_args()
    options = vars(args)
    print(options)

    assert args.dataset in BenchmarkMapping.keys(), "Required dataset is not implemented!"

    run_experiment(args)
