from collections import OrderedDict

import numpy as np

from .graph_dense.graph_dense import GraphHyperparameter


class SearchSpace:
    def __init__(self, *hyperparameters):
        self._num_hps = len(hyperparameters)
        self._hyperparameters = OrderedDict()
        self._hps = []
        self._graphs = []

        for hyperparameter in hyperparameters:
            self._hyperparameters[hyperparameter.name] = hyperparameter
            if isinstance(hyperparameter, GraphHyperparameter):
                self._graphs.append(hyperparameter)
            else:
                self._hps.append(hyperparameter)

        self.name = ""

    def sample(self):
        for hyperparameter in self._hyperparameters.values():
            hyperparameter.sample()

        self.name = self.parse()

    def mutate(self, config=None, mutate_probability_per_hyperparameter=1.0, patience=50):
        new_config = []
        for hyperparameter in self._hyperparameters.values():
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
        config = ""
        for hp in self._hyperparameters.values():
            config += (
                hp.name
                if isinstance(hp, GraphHyperparameter)
                else "{}-{}".format(hp.name, hp.value)
            )
            config += "_"
        return config

    @property
    def id(self):
        return self.name

    def get_graphs(self):
        return [graph.value for graph in self._graphs]

    def get_hps(self):
        return [hp.value for hp in self._hps]

    def get_hyperparameter_by_name(self, name: str):
        hp = self._hyperparameters.get(name)

        if hp is None:
            raise KeyError("Hyperparameter is not a part of the search space!")

        return hp
