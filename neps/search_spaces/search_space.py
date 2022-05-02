from __future__ import annotations

import collections.abc
import pprint
import random
from collections import OrderedDict
from copy import deepcopy

import ConfigSpace as CS
import numpy as np

from . import (
    CategoricalParameter,
    ConstantParameter,
    FloatParameter,
    IntegerParameter,
    NumericalParameter,
)
from .graph_grammar.graph import Graph
from .parameter import Parameter


def pipeline_space_from_configspace(
    configspace: CS.ConfigurationSpace,
) -> dict[str, Parameter]:
    pipeline_space = dict()
    parameter: Parameter
    for hyperparameter in configspace.get_hyperparameters():
        if isinstance(hyperparameter, CS.CategoricalHyperparameter):
            parameter = CategoricalParameter(hyperparameter.choices)
        elif isinstance(hyperparameter, CS.OrdinalHyperparameter):
            parameter = CategoricalParameter(hyperparameter.sequence)
        elif isinstance(hyperparameter, CS.UniformIntegerHyperparameter):
            parameter = IntegerParameter(
                lower=hyperparameter.lower,
                upper=hyperparameter.upper,
                log=hyperparameter.log,
            )
        elif isinstance(hyperparameter, CS.UniformFloatHyperparameter):
            parameter = FloatParameter(
                lower=hyperparameter.lower,
                upper=hyperparameter.upper,
                log=hyperparameter.log,
            )
        else:
            raise ValueError(f"Unknown hyperparameter type {hyperparameter}")
        pipeline_space[hyperparameter.name] = parameter
    return pipeline_space


class SearchSpace(collections.abc.Mapping):
    def __init__(self, **hyperparameters):
        self._num_hps = len(hyperparameters)
        self.hyperparameters = OrderedDict()
        self._hps = []
        self._graphs = []

        self.fidelity = None
        self.has_prior = False
        for key, hyperparameter in hyperparameters.items():
            self.hyperparameters[key] = hyperparameter

            if isinstance(hyperparameter, NumericalParameter):
                self._hps.append(hyperparameter)
            else:
                self._graphs.append(hyperparameter)

            # Only integer / float parameters can be fidelities, so check these
            if hasattr(hyperparameter, "is_fidelity") and hyperparameter.is_fidelity:
                if self.fidelity is not None:
                    raise ValueError(
                        "neps only supports one fidelity parameter in the pipeline space,"
                        " but multiple were given. (Hint: check you pipeline space for "
                        "multiple is_fidelity=True)"
                    )
                self.fidelity = hyperparameter

            # Check if defaults exists to construct prior from
            if hasattr(hyperparameter, "default") and hyperparameter.default is not None:
                self.has_prior = True
            elif hasattr(hyperparameter, "has_prior") and hyperparameter.has_prior:
                self.has_prior = True

    def compute_prior(self, log: bool = False):
        density_value = 0.0 if log else 1.0
        for hyperparameter in self.hyperparameters.values():
            if hyperparameter.has_prior:
                if log:
                    density_value += hyperparameter.compute_prior(log=True)
                else:
                    density_value *= hyperparameter.compute_prior(log=False)
        return density_value

    def has_fidelity(self):
        return self.fidelity is not None

    def sample(self, user_priors: bool = False, patience: int = 1) -> SearchSpace:
        sample = self.copy()
        for hp_name, hyperparameter in sample.hyperparameters.items():
            if (
                isinstance(hyperparameter, NumericalParameter)
                and hyperparameter.is_fidelity
            ):
                continue
            for _ in range(patience):
                try:
                    hyperparameter.sample(user_priors=user_priors)
                    break
                except ValueError:
                    pass
            else:
                raise ValueError(
                    f"Could not sample valid config for {hp_name} in {patience} tries!"
                )
        return sample

    def mutate(
        self,
        config=None,  # pylint: disable=unused-argument
        patience=50,
        mutation_strategy="smbo",
    ):

        if mutation_strategy == "smbo":
            new_config = self._smbo_mutation(patience)
        else:
            raise NotImplementedError("No such mutation strategy!")

        child = SearchSpace(**dict(zip(self.hyperparameters.keys(), new_config)))

        return child

    def _smbo_mutation(self, patience=50):
        new_config = self.get_array()

        for _ in range(patience):
            idx = random.randint(0, len(new_config) - 1)
            hp = new_config[idx]
            if isinstance(hp, NumericalParameter) and hp.is_fidelity:
                continue
            try:
                new_config[idx] = hp.mutate()
                break
            except Exception:
                continue
        return new_config

    def crossover(
        self,
        config2,
        crossover_probability_per_hyperparameter: float = 1.0,
        patience: int = 50,
        crossover_strategy: str = "simple",
    ):

        if crossover_strategy == "simple":
            new_config1, new_config2 = self._simple_crossover(
                config2, crossover_probability_per_hyperparameter, patience
            )
        else:
            raise NotImplementedError("No such crossover strategy!")

        if len(self.hyperparameters.keys()) != len(new_config1):
            raise Exception("Cannot crossover")
        child1 = SearchSpace(**dict(zip(self.hyperparameters.keys(), new_config1)))
        child2 = SearchSpace(**dict(zip(self.hyperparameters.keys(), new_config2)))
        return child1, child2

    def _simple_crossover(
        self,
        config2,
        crossover_probability_per_hyperparameter: float = 1.0,
        patience: int = 50,
    ):
        new_config1 = []
        new_config2 = []
        for key, hyperparameter in self.hyperparameters.items():
            if (
                hasattr(hyperparameter, "crossover")
                and np.random.random() < crossover_probability_per_hyperparameter
            ):
                while patience > 0:
                    try:
                        child1, child2 = hyperparameter.crossover(
                            config2.hyperparameters[key]
                        )
                        new_config1.append(child1)
                        new_config2.append(child2)
                        break
                    except NotImplementedError:
                        new_config1.append(hyperparameter)
                        new_config2.append(config2.hyperparameters[key])
                        break
                    except Exception:
                        patience -= 1
                        continue
            else:
                new_config1.append(hyperparameter)
                new_config2.append(config2.hyperparameters[key])

        return new_config1, new_config2

    def get_graphs(self):
        return [graph.value for graph in self._graphs]

    def get_hps(self):
        # Numerical hyperparameters are split into:
        # - categorical HPs
        # - float/integer continuous HPs
        # user defined dimensionality split not supported yet!
        cont_hps = []
        cat_hps = []

        for hp in self._hps:
            hp_value = hp.normalized().value
            if isinstance(hp, CategoricalParameter):
                cat_hps.append(hp_value)
            else:
                cont_hps.append(hp_value)
        return {
            "continuous": None if len(cont_hps) == 0 else cont_hps,
            "categorical": None if len(cat_hps) == 0 else cat_hps,
        }

    def get_array(self):
        return list(self.hyperparameters.values())

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
        hps = self.get_hps()
        if all(hp is None for hp in hps.values()):
            return None
        for k, v in hps.items():
            d[k] = 0 if v is None else len(v)
        return d

    def serialize(self):
        return {key: hp.serialize() for key, hp in self.hyperparameters.items()}

    def load_from(self, config: dict):
        self._hps = []
        self._graphs = []
        for name in config.keys():
            self.hyperparameters[name].load_from(config[name])
            if isinstance(self.hyperparameters[name], NumericalParameter):
                self._hps.append(self.hyperparameters[name])
            else:
                self._graphs.append(self.hyperparameters[name])

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, key):
        hp = self.hyperparameters[key]
        if isinstance(hp, Graph) or hp.value is None:
            return hp
        else:
            return hp.value

    def __iter__(self):
        return iter(self.hyperparameters)

    def __len__(self):
        return len(self.hyperparameters)

    def __str__(self):
        return pprint.pformat(self.hyperparameters)
