import random
from collections import OrderedDict

import numpy as np

from . import HyperparameterMapping


class SearchSpace:
    def __init__(self, *hyperparameters):
        self._num_hps = len(hyperparameters)
        self._hyperparameters = OrderedDict()
        self._hps = []
        self._graphs = []

        for hyperparameter in hyperparameters:
            if "hp_name" in dir(hyperparameter):
                self._hyperparameters[hyperparameter.hp_name] = hyperparameter
            else:
                self._hyperparameters[hyperparameter.name] = hyperparameter
            if (
                isinstance(hyperparameter, HyperparameterMapping["graph_dense"])
                or isinstance(hyperparameter, HyperparameterMapping["graph_grammar"])
                or isinstance(
                    hyperparameter, HyperparameterMapping["graph_grammar_repetitive"]
                )
            ):
                self._graphs.append(hyperparameter)
            else:
                self._hps.append(hyperparameter)

        self.name = ""

    def sample(self):
        for hyperparameter in self._hyperparameters.values():
            hyperparameter.sample()

        self.name = self.parse()

    def mutate(
        self,
        config=None,
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

        child = SearchSpace(*new_config)
        child.name = child.parse()

        return child

    def _simple_mutation(self, mutate_probability_per_hyperparameter=1.0, patience=50):
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

    def parse(self):
        config = ""
        for hp in self._hyperparameters.values():
            if isinstance(hp, HyperparameterMapping["graph_dense"]):
                config += hp.name
            elif isinstance(hp, HyperparameterMapping["graph_grammar"]) or isinstance(
                hp, HyperparameterMapping["graph_grammar_repetitive"]
            ):
                config += hp.id
            else:
                config += f"{hp.name}-{hp.value}"
            config += "_"
        return config

    @property
    def id(self):
        return self.name

    def get_graphs(self):
        return [graph.value for graph in self._graphs]

    def get_hps(self):
        return [hp.value for hp in self._hps]

    def get_array(self):
        return list(self._hyperparameters.values())

    def get_hyperparameter_by_name(self, name: str):
        hp = self._hyperparameters.get(name)

        if hp is None:
            raise KeyError("Hyperparameter is not a part of the search space!")

        return hp

    def transform(self):
        [hp._transform() for hp in self.get_array()]

    def inv_transform(self):
        [hp._inv_transform() for hp in self.get_array()]

    def get_dictionary(self):
        d = dict()
        for hp in self._hyperparameters.values():
            d = {**d, **hp.get_dictionary()}
        return d

    def create_from_id(self, config):
        self._hps = []
        self._graphs = []
        for name in config.keys():
            self._hyperparameters[name].create_from_id(config[name])
            if (
                isinstance(
                    self._hyperparameters[name], HyperparameterMapping["graph_dense"]
                )
                or isinstance(
                    self._hyperparameters[name], HyperparameterMapping["graph_grammar"]
                )
                or isinstance(
                    self._hyperparameters[name],
                    HyperparameterMapping["graph_grammar_repetitive"],
                )
            ):
                self._graphs.append(self._hyperparameters[name])
            else:
                self._hps.append(self._hyperparameters[name])

        self.name = self.parse()
