import random
from collections import OrderedDict

import numpy as np

from . import CategoricalParameter, ConstantParameter, NumericalParameter


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

        child = SearchSpace(**dict(zip(self.hyperparameters.keys(), new_config)))

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
                new_config[idx] = hp.mutate()
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
        # Numerical hyperparameters are split into:
        # - categorical HPs
        # - float/integer continuous HPs
        # user defined dimensionality split not supported yet!
        cont_hps = []
        cat_hps = []
        for hp in self._hps:
            if isinstance(hp, CategoricalParameter):
                cat_hps.append(hp.value)
            else:
                cont_hps.append(hp.value)
        return {
            "continuous": None if len(cont_hps) == 0 else cont_hps,
            "categorical": None if len(cat_hps) == 0 else cat_hps,
        }

    def get_array(self):
        return list(self.hyperparameters.values())

    def get_dictionary(self):
        return dict(zip(self.hyperparameters.keys(), self.id))

    def create_from_id(self, config: dict):
        self._hps = []
        self._graphs = []
        for name in config.keys():
            self.hyperparameters[name].create_from_id(config[name])
            if isinstance(self.hyperparameters[name], NumericalParameter):
                self._hps.append(self.hyperparameters[name])
            else:
                self._graphs.append(self.hyperparameters[name])

    def add_constant_hyperparameter(self, value=None):
        if value is not None:
            hp = ConstantParameter(value=value)
        else:
            raise NotImplementedError("Adding hps is supported only by value")
        self._add_hyperparameter(hp)

    def _add_hyperparameter(self, hp=None):
        self.hyperparameters[str(self._num_hps)] = hp
        if isinstance(hp, NumericalParameter):
            self._hps.append(hp)
        else:
            self._graphs.append(hp)
        self._num_hps += 1

    def get_vectorial_dim(self):
        # search space object may contain either continuous or categorical hps
        d = {}
        for k, v in self.get_hps().items():
            d[k] = 0 if v is None else len(v)
        return d
