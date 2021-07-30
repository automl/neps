import random

from ..core.optimizer import Optimizer


class RandomSearch(Optimizer):
    def __init__(self, args, objective):
        super().__init__(args, objective)
        self.sampled_idx = []

    def initialize_model(self, **kwargs):
        pass

    def update_model(self, **kwargs):
        pass

    def propose_new_location(self, **kwargs):
        pool = kwargs["pool"]
        next_x = random.sample(pool, self.batch_size)
        self.sampled_idx.append(next_x)

        next_graphs = tuple()
        next_hps = tuple()
        for graph, hp in next_x:
            next_graphs += (graph,)
            next_hps += (hp,)

        return next_graphs, next_hps
