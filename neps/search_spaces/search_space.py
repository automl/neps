from __future__ import annotations

import collections.abc
import pprint
import random
from collections import OrderedDict
from copy import deepcopy

import ConfigSpace as CS
import metahyper
import numpy as np

from ..utils.common import has_instance
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


class SearchSpace(collections.abc.Mapping, metahyper.api.Configuration):
    def __init__(self, **hyperparameters):
        self.hyperparameters = OrderedDict()

        self.fidelity = None
        self.has_prior = False
        for key, hyperparameter in hyperparameters.items():
            self.hyperparameters[key] = hyperparameter

            # Only integer / float parameters can be fidelities, so check these
            if hyperparameter.is_fidelity:
                if self.fidelity is not None:
                    raise ValueError(
                        "neps only supports one fidelity parameter in the pipeline space,"
                        " but multiple were given. (Hint: check you pipeline space for "
                        "multiple is_fidelity=True)"
                    )
                elif not isinstance(hyperparameter, FloatParameter):
                    raise ValueError(
                        "neps only suport float and integer fidelity parameters"
                    )
                self.fidelity = hyperparameter

            # Check if defaults exists to construct prior from
            if hasattr(hyperparameter, "default") and hyperparameter.default is not None:
                self.has_prior = True
            elif hasattr(hyperparameter, "has_prior") and hyperparameter.has_prior:
                self.has_prior = True

    @property
    def has_fidelity(self):
        return self.fidelity is not None

    def compute_prior(self, log: bool = False):
        density_value = 0.0 if log else 1.0
        for hyperparameter in self.hyperparameters.values():
            if hyperparameter.has_prior:
                if log:
                    density_value += hyperparameter.compute_prior(log=True)
                else:
                    density_value *= hyperparameter.compute_prior(log=False)
        return density_value

    def sample(self, user_priors: bool = False, patience: int = 1) -> SearchSpace:
        sample = self.copy()
        for hp_name, hyperparameter in sample.hyperparameters.items():
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

        child = SearchSpace(**new_config)

        return child

    def _smbo_mutation(self, patience=50):
        new_config = deepcopy(self.hyperparameters)
        config_hp_names = list(new_config)

        for _ in range(patience):
            idx = random.randint(0, len(new_config) - 1)
            hp_name = config_hp_names[idx]
            hp = new_config[hp_name]
            if isinstance(hp, NumericalParameter) and hp.is_fidelity:
                continue
            try:
                new_config[hp_name] = hp.mutate()
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
                and not hyperparameter.is_fidelity
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

    def get_normalized_hp_categories(self):
        hps = {
            "continuous": [],
            "categorical": [],
            "graphs": [],
        }
        for hp in self.values():
            hp_value = hp.normalized().value
            if isinstance(hp, CategoricalParameter):
                hps["categorical"].append(hp_value)
            elif isinstance(hp, NumericalParameter):
                hps["continuous"].append(hp_value)
            else:
                hps["graphs"].append(hp_value)
        return hps

    def hp_values(self):
        return {
            hp_name: hp if isinstance(hp, Graph) else hp.value
            for hp_name, hp in self.hyperparameters.items()
        }

    def split_hps_fidelities(self):
        hps, fidelities = {}, {}
        for hp_name, hp in self.items():
            if hp.is_fidelity:
                fidelities[hp_name] = deepcopy(hp)
            else:
                hps[hp_name] = deepcopy(hp)
        return SearchSpace(**hps), SearchSpace(**fidelities)

    def add_constant_hyperparameter(self, name=None, value=None):
        if value is not None:
            hp = ConstantParameter(value=value)
        else:
            raise NotImplementedError("Adding hps is supported only by value")
        self._add_hyperparameter(name, hp)

    def _add_hyperparameter(self, name=None, hp=None):
        if name is None:
            id_new_hp = len(self.hyperparameters)
            while str(id_new_hp) in self.hyperparameters:
                id_new_hp += 1
        else:
            id_new_hp = name
        self.hyperparameters[str(id_new_hp)] = hp

    def get_vectorial_dim(self):
        if not has_instance(self.values(), NumericalParameter):
            return None
        features = {"continuous": 0, "categorical": 0}
        for hp in self.values():
            if isinstance(hp, CategoricalParameter):
                features["categorical"] += 1
            elif isinstance(hp, NumericalParameter):
                features["continuous"] += 1
        return features

    def set_to_max_fidelity(self):
        self.fidelity.value = self.fidelity.upper

    def serialize(self):
        return {key: hp.serialize() for key, hp in self.hyperparameters.items()}

    def load_from(self, config: dict):
        for name in config.keys():
            self.hyperparameters[name].load_from(config[name])

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, key):
        return self.hyperparameters[key]

    def __iter__(self):
        return iter(self.hyperparameters)

    def __len__(self):
        return len(self.hyperparameters)

    def __str__(self):
        return pprint.pformat(self.hyperparameters)
