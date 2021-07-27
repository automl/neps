import argparse

from nasbowl.kernel_operators import *
import nasbowl.models
from nasbowl.acqusition_functions import *

from nasbowl.initial_design.generate_test_graphs import *
from nasbowl.initial_design.init_random_uniform import *
from nasbowl.benchmarks.nas.nasbench301 import NASBench301
from nasbowl.benchmarks.nas.nasbench201 import NASBench201
from nasbowl.benchmarks.hpo.branin2 import Branin2
from nasbowl.benchmarks.hpo.hartmann3 import Hartmann3
from nasbowl.benchmarks.hpo.hartmann6 import Hartmann6
from nasbowl.benchmarks.hpo.counting_ones import CountingOnes

from utils.util import StatisticsTracker
from optimizer import RandomSearch, BayesianOptimization
from sampler import Sampler

import warnings

warnings.simplefilter("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description='CNAS')
parser.add_argument('--dataset', default='nasbench301', help='The benchmark dataset to run the experiments. '
                                                             'options = ["nasbench201", "nasbench301", "hartmann3", '
                                                             '"hartmann6", "counting_ones"].')
parser.add_argument('--n_repeat', type=int, default=1, help='number of repeats of experiments')
parser.add_argument('--n_init', type=int, default=40, help='number of initialising points')
parser.add_argument("--max_iters", type=int, default=100, help='number of iterations for the search')
parser.add_argument('-ps', '--pool_size', type=int, default=100,
                    help='number of candidates generated at each iteration')
parser.add_argument('--mutate_size', type=int, help='number of mutation candidates. By default, half of the pool_size '
                                                    'is generated from mutation.')
parser.add_argument('--pool_strategy', default='random', help='the pool generation strategy. Options: random,'
                                                              'mutate')
parser.add_argument('--save_path', default='results/', help='path to save log file')
parser.add_argument('-s', '--strategy', default='gbo', help='optimisation strategy: option: gbo (graph bo), '
                                                            'random (random search)')
parser.add_argument('-a', "--acquisition", default='UCB', help='the acquisition function for the BO algorithm. option: '
                                                               'UCB, EI, AEI')
parser.add_argument('-kh', '--hp_kernel', default='m52',
                    help='hp kernel to use. Can be [rbf, m52, m32]')
parser.add_argument('-dsk', '--domain_se_kernel', default='m52',
                    help='Successive Embedding kernel on the domain to use. Can be [rbf, m52, m32]')
# parser.add_argument('-p', '--plot', action='store_true', help='whether to plot the procedure each iteration.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of samples to evaluate')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--optimize_arch', action='store_true', help='Whether to optimize arch')
parser.add_argument('--optimize_hps', action='store_true', help='Whether to optimize hps')
parser.add_argument('--cuda', action='store_true', help='Whether to use GPU acceleration')
# parser.add_argument('--mutate_unpruned_archs', action='store_true',
#                     help='Whether to mutate on the unpruned archs. This option is only valid if mutate '
#                          'is specified as the pool_strategy')
parser.add_argument('--no_isomorphism', action='store_true', help='Whether to allow mutation to return'
                                                                  'isomorphic architectures')
parser.add_argument('--maximum_noise', default=0.01, type=float, help='The maximum amount of GP jitter noise variance')
parser.add_argument('--log', action='store_true', help='Whether to report the results in log scale')
args = parser.parse_args()


if args.cuda and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Initialise the objective function. Negative ensures a maximisation task that is assumed by the acquisition
# function.
assert args.dataset in ['nasbench201', 'nasbench301', 'branin2', 'hartmann3', 'hartmann6', 'counting_ones'],\
    "Required dataset " + args.dataset + " is not implemented!"
if args.dataset == 'nasbench201':
    objective = NASBench201(data_dir='nasbowl/benchmarks/nas/nb_configfiles/data',
                            log_scale=args.log, negative=True, seed=args.seed)
elif args.dataset == 'nasbench301':
    objective = NASBench301(log_scale=args.log, negative=True, seed=args.seed)
elif args.dataset == 'branin2':
    objective = Branin2(log_scale=args.log, negative=True, seed=args.seed)
elif args.dataset == 'hartmann3':
    objective = Hartmann3(log_scale=args.log, negative=False, seed=args.seed)
elif args.dataset == 'hartmann6':
    objective = Hartmann6(log_scale=args.log, negative=False, seed=args.seed)
elif args.dataset == 'counting_ones':
    objective = CountingOnes(log_scale=args.log, negative=False, seed=args.seed)
else:
    objective = None
# Initialise the optimizer strategy
assert args.strategy in ['random', 'gbo']
if args.strategy == 'random':
    optimizer = RandomSearch(args, objective)
elif args.strategy == 'gbo':
    optimizer = BayesianOptimization(args, objective)
else:
    optimizer = None
assert args.pool_strategy in ['random', 'mutate', ]
sampler = Sampler(args, objective)

experiments = StatisticsTracker(args)

for seed in range(args.seed, args.seed + args.n_repeat):
    experiments.reset(seed)

    # Take n_init random samples
    x_graphs, x_hps = sampler.sample(pool_size=args.n_init)

    # & evaluate
    y_np_list = [objective.eval(graphs_, hps_) for graphs_, hps_ in list(zip(x_graphs, x_hps))]
    y = torch.tensor([y[0] for y in y_np_list]).float()
    train_details = [y[1] for y in y_np_list]

    # Initialise the GP surrogate
    optimizer.initialize_model(x_graphs=deepcopy(x_graphs), x_hps=deepcopy(x_hps), y=deepcopy(y))

    # Main optimization loop

    while experiments.has_budget():

        pool_graphs, pool_hps = sampler.sample(args.pool_size)

        # Propose new location to evaluate
        query_dict = {'iters': experiments.iteration, 'pool': list(zip(pool_graphs, pool_hps))}
        next_graphs, next_hps = optimizer.propose_new_location(**query_dict)

        # Evaluate this location from the objective function
        detail = [objective.eval(graphs_, hps_) for graphs_, hps_ in list(zip(next_graphs, next_hps))]
        next_y = [y[0] for y in detail]
        train_details += [y[1] for y in detail]

        if optimizer.surrogate_model is not None:
            pool_graphs.extend(next_graphs)
            pool_hps.extend(next_hps)

        x_graphs.extend(next_graphs)
        x_hps.extend(next_hps)
        y = torch.cat((y, torch.tensor(next_y).view(-1))).float()

        # Update the GP Surrogate
        optimizer.update_model(x_graphs=deepcopy(x_graphs), x_hps=deepcopy(x_hps), y=deepcopy(y))
        experiments.print(list(zip(x_graphs, x_hps)), y, next_y, train_details)

    incumbent_config = []
    for inc_graphs, inc_hps in experiments.incumbents:
        if isinstance(objective, NASBench301):
            incumbent_config.append(objective.tuple_to_config_dict(inc_graphs, inc_hps))
        elif isinstance(objective, NASBench201):
            if args.optimize_arch:
                incumbent_config.append(list(zip(inc_graphs, inc_hps)))
            else:
                incumbent_config.append(inc_hps)
        else:
            incumbent_config.append(inc_hps)

    experiments.save_results(incumbents_config=incumbent_config)



