from __future__ import annotations

import collections.abc
import pprint
import random
from collections import OrderedDict
from copy import copy
from itertools import accumulate
from typing import Any

import ConfigSpace as CS
import metahyper
import numpy as np
from more_itertools import last

from ..utils.common import disabled
from .graph_grammar.graph import Graph
from .graph_grammar.graph_grammar import GraphGrammar
from .numerical.categorical import CategoricalParameter
from .numerical.constant import ConstantParameter
from .numerical.float import FloatParameter
from .numerical.integer import IntegerParameter
from .numerical.numerical import NumericalParameter
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

        child = self.copy()
        child.hyperparameters.update(new_config)

        return child

    def _smbo_mutation(self, patience=50):
        new_config = self.copy().hyperparameters
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
        child1, child2 = self.copy(), self.copy()
        child1.hyperparameters.update(
            **dict(zip(self.hyperparameters.keys(), new_config1))
        )
        child2.hyperparameters.update(
            **dict(zip(self.hyperparameters.keys(), new_config2))
        )
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
            "constants": [],
        }
        for hp in self.values():
            hp_value = hp.normalized().value
            if isinstance(hp, CategoricalParameter):
                hps["categorical"].append(hp_value)
            elif isinstance(hp, FloatParameter):
                hps["continuous"].append(hp_value)
            elif isinstance(hp, GraphGrammar):
                hps["graphs"].append(hp_value)
            elif isinstance(hp, ConstantParameter):
                hps["constants"].append(hp)
            else:
                raise ValueError(f"Unknown category for {hp} ({hp.__class__})")
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
                fidelities[hp_name] = hp.copy()
            else:
                hps[hp_name] = hp.copy()
        return hps, fidelities

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

    def set_to_max_fidelity(self):
        self.fidelity.value = self.fidelity.upper

    def serialize(self):
        return {key: hp.serialize() for key, hp in self.hyperparameters.items()}

    def load_from(self, config: dict):
        for name in config.keys():
            self.hyperparameters[name].load_from(config[name])

    def copy(self):
        """Returns the shallowest copy of the same space, creating a new space
        without copying the objects referenced by the space or the hyperparameters."""
        new_space = copy(self)
        for hp_name, hp in new_space.items():
            new_space.hyperparameters[hp_name] = hp.copy()
        return new_space

    def __getitem__(self, key):
        return self.hyperparameters[key]

    def __iter__(self):
        return iter(self.hyperparameters)

    def __len__(self):
        return len(self.hyperparameters)

    def __str__(self):
        return pprint.pformat(self.hyperparameters)


class SparseSearchSpace(SearchSpace):
    """Represents a search space where the values that can actually be evaluated
    are sparse points across the space."""

    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)
        self.allowed_configs: list[dict[str, Any]] = []
        self.allowed_config_probas = []

    def sample_configs_from(self, allowed_configs: list[dict[str, Any]]):
        """Set the set of allowed configurations. For each hyperparameter,
        the value should be a Parameter, or a serialized value.
        """
        allowed_config_spaces = []
        allowed_config_probas = []
        print("Looping over cfgs")
        hp_copies = {hp_name: hp.copy() for hp_name, hp in self.items()}
        for cfg in allowed_configs:
            if set(cfg.keys()) != set(self.keys()):
                raise ValueError(
                    f"The configuration {cfg} doesn't match"
                    f"the set of hyperparameters of this SearchSpace {list(self.keys())}"
                )

            cfg, cfg_proba = {**cfg}, 1.0
            for hp_name, hp in cfg.items():
                if isinstance(hp, Parameter):
                    assert type(hp) is type(self[hp_name])
                    cfg[hp_name] = hp.copy()
                    cfg_proba *= hp.prior_probability()
                else:
                    hp_copies[hp_name].load_from(hp)
                    cfg_proba *= hp_copies[hp_name].prior_probability()
                assert cfg_proba >= 0, "Can't have a prior probability of 0"

            allowed_config_spaces.append(cfg)
            allowed_config_probas.append(cfg_proba)

        print("done")
        proba_sum = sum(allowed_config_probas)
        self.allowed_config_probas = [0.0] + list(
            map(lambda p: p / proba_sum, accumulate(allowed_config_probas))
        )[:-1]
        self.allowed_configs = allowed_config_spaces

    def sample(self, user_priors: bool = False, patience: int = 1) -> SearchSpace:
        if not self.allowed_configs:
            raise ValueError(
                "The SparseSearchSpace hasn't been initialized by"
                "calling sample_configs_from"
            )
        cfg_cursor = random.random()
        if user_priors:
            choosen_cfg, _ = last(
                filter(
                    lambda cfg_p: cfg_p[1] <= cfg_cursor,
                    zip(self.allowed_configs, self.allowed_config_probas),
                )
            )
        else:
            choosen_cfg = random.choice(self.allowed_configs)

        sample = self.copy()
        for hp_name in sample:
            if isinstance(choosen_cfg[hp_name], Parameter):
                sample.hyperparameters[hp_name] = choosen_cfg[hp_name].copy()
            else:
                sample[hp_name].load_from(choosen_cfg[hp_name])
        return sample

    mutate = disabled
    crossover = disabled
