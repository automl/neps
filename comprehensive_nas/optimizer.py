import networkx as nx
import ConfigSpace

from nasbowl.kernel_operators import *
import nasbowl.models
from nasbowl.acqusition_functions import *

from nasbowl.initial_design.generate_test_graphs import *
from nasbowl.initial_design.init_random_uniform import *
from nasbowl.benchmarks.nas.nasbench301 import *
# from nasbowl.benchmarks.nas.nasbench201 import NASBench201


class Optimizer:
    def __init__(self, args, objective):
        self.surrogate_model = None
        self.verbose = args.verbose
        self.maximum_noise = args.maximum_noise
        self.strategy = args.strategy
        self.batch_size = args.batch_size

    def generate_new_pool(self):
        pass

    def evaluate(self):
        pass

    def initialize_model(self, **kwargs):
        raise NotImplementedError

    def update_model(self, **kwargs):
        raise NotImplementedError

    def propose_new_location(self, **kwargs):
        raise NotImplementedError


class RandomSearch(Optimizer):
    def __init__(self, args, objective):
        super(RandomSearch, self).__init__(args, objective)
        self.sampled_idx = []

    def initialize_model(self, **kwargs):
        pass

    def update_model(self, **kwargs):
        pass

    def propose_new_location(self, **kwargs):
        pool = kwargs['pool']
        next_x = random.sample(pool, self.batch_size)
        self.sampled_idx.append(next_x)

        next_graphs = tuple()
        next_hps = tuple()
        for graph, hp in next_x:
            next_graphs += (graph,)
            next_hps += (hp,)

        return next_graphs, next_hps


class BayesianOptimization(Optimizer):
    def __init__(self, args, objective):
        super(BayesianOptimization, self).__init__(args, objective)
        self.hp_kernel = args.hp_kernel
        self.domain_se_kernel = args.domain_se_kernel
        self.acquisition = args.acquisition

    def initialize_model(self, x_graphs, x_hps, y):
        kern = []
        if None not in x_graphs:
            n_graph_kernels = len(x_graphs[0]) if isinstance(x_graphs[0], tuple) else 1
            for _ in range(n_graph_kernels):
                # Graph kernel_operators
                kern.append(GraphKernelMapping['wl'](se_kernel=StationaryKernelMapping[self.domain_se_kernel]))

        self.surrogate_model = nasbowl.models.ComprehensiveGP(train_x_graphs=x_graphs,
                                                              train_x_hps=x_hps,
                                                              train_y=y,
                                                              graph_kernels=kern,
                                                              hp_cont_kernel=StationaryKernelMapping[self.hp_kernel],
                                                              verbose=self.verbose)
        self.surrogate_model.fit(wl_subtree_candidates=tuple(range(1, 4)) if None not in x_graphs
                                 else tuple(),
                                 max_lik=self.maximum_noise)

    def update_model(self, x_graphs, x_hps, y):

        self.surrogate_model.reset_XY(x_graphs, x_hps, y)
        self.surrogate_model.fit(wl_subtree_candidates=tuple(range(1, 4)),
                                 optimize_lik=True,
                                 max_lik=self.maximum_noise
                                 )

    def propose_new_location(self, **kwargs):
        iters = kwargs['iters']
        pool = kwargs['pool']

        # Init the acquisition function
        if self.acquisition in AcquisitionMapping.keys():
            a = AcquisitionMapping[self.acquisition](surrogate_model=self.surrogate_model, strategy=self.strategy,
                                                     iters=iters)
        else:
            raise ValueError("Acquisition function" + str(self.acquisition) + ' is not understood!')

        # Ask for a location proposal from the acquisition function..
        next_x, eis, indices = a.propose_location(top_n=self.batch_size, candidates=pool.copy())

        next_graphs = tuple()
        next_hps = tuple()
        for graph, hp in next_x:
            next_graphs += (graph,)
            next_hps += (hp,)

        return next_graphs, next_hps
