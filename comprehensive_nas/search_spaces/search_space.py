import itertools

import numpy as np

from .graph_dense.graph_dense import GraphHyperparameter


class SearchSpace:
    def __init__(self, *hyperparameters):
        self.hps = []
        self.graphs = []

        for hyperparameter in hyperparameters:
            if isinstance(hyperparameter, GraphHyperparameter):
                self.graphs.append(hyperparameter)
            else:
                self.hps.append(hyperparameter)

        self.name = ""

    def sample_config(self):
        for hyperparameter in set(itertools.chain.from_iterable((self.hps, self.graphs))):
            hyperparameter.sample()

        self.name = self.parse()

    def mutate(self, config=None, mutate_probability_per_hyperparameter=1.0, patience=50):
        new_config = []
        for hyperparameter in set(itertools.chain.from_iterable((self.hps, self.graphs))):
            if np.random.random() < mutate_probability_per_hyperparameter:
                while patience > 0:
                    try:
                        new_config.append(hyperparameter.mutate())
                        break
                    except Exception:
                        patience -= 1
                        continue
            else:
                new_config.append(hyperparameter)
        child = SearchSpace(*new_config)
        child.name = child.parse()
        return child

    def parse(self):
        config = []
        if self.graphs is not None:
            if not isinstance(self.graphs, list):
                # Convert to a singleton list
                gl = [self.graphs]
            else:
                gl = self.graphs
            config.append([g.name for g in gl])
        if self.hps is not None:
            config.append(["{}-{}".format(hp.name, hp.value) for hp in self.hps])
        return config

    @property
    def id(self):
        return self.name

    def get_graph(self):
        return self.graphs

    def get_hps(self):
        return self.hps
