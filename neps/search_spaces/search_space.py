import random
from collections import OrderedDict
from typing import List

import numpy as np

from . import NumericalParameter


class SearchSpace:
    def __init__(self, **hyperparameters):
        self._num_hps = len(hyperparameters)
        self.hyperparameters = OrderedDict()
        self._hps = []
        self._graphs = []

        for key, hyperparameter in hyperparameters.items():
            self.hyperparameters[key] = hyperparameter

            if isinstance(hyperparameter, NumericalParameter):
                self._hps.append(hyperparameter)
            else:
                self._graphs.append(hyperparameter)

    def sample(self):
        for hyperparameter in self.hyperparameters.values():
            hyperparameter.sample()

    def mutate(
        self,
        config=None,  # pylint: disable=unused-argument
        mutate_probability_per_hyperparameter=1.0,
        patience=50,
        mutation_strategy="simple",
    ):

        if mutation_strategy == "simple":
            new_config = self._simple_mutation(
                mutate_probability_per_hyperparameter, patience
            )
        elif mutation_strategy == "smbo":
            new_config = self._smbo_mutation(patience)
        else:
            raise NotImplementedError("No such mutation strategy!")

        child = SearchSpace(dict(zip(self.hyperparameters.keys(), new_config)))

        return child

    def _simple_mutation(self, mutate_probability_per_hyperparameter=1.0, patience=50):
        new_config = []
        for hyperparameter in self.hyperparameters.values():
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

        return new_config

    def _smbo_mutation(self, patience=50):
        new_config = self.get_array()
        idx = random.randint(0, self._num_hps - 1)
        hp = new_config[idx]

        while patience > 0:
            try:
                new_config[idx] = hp.mutate(mutation_strategy="local_search")
                break
            except Exception:
                patience -= 1
                continue
        return new_config

    def get_graphs(self):
        return [graph.value for graph in self._graphs]

    @property
    def id(self):
        return [hp.id for hp in self.get_array()]

    def get_hps(self):
        return [hp.value for hp in self._hps]

    def get_array(self):
        return list(self.hyperparameters.values())

    def get_dictionary(self):
        return dict(zip(self.hyperparameters.keys(), self.id))

    def create_from_id(self, config: List[str]):
        self._hps = []
        self._graphs = []
        for name in config.keys():
            self.hyperparameters[name].create_from_id(config[name])
            if isinstance(self.hyperparameters[name], NumericalParameter):
                self._hps.append(self.hyperparameters[name])
            else:
                self._graphs.append(self.hyperparameters[name])
