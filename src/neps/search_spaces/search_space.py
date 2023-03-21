from __future__ import annotations

import collections.abc
import pprint
import random
from collections import OrderedDict
from copy import deepcopy
from itertools import product

import ConfigSpace as CS
import numpy as np

from ..utils.common import has_instance
from . import (
    CategoricalParameter,
    ConstantParameter,
    FloatParameter,
    IntegerParameter,
    NumericalParameter,
)
from .architecture.graph import Graph
from .parameter import Parameter


def pipeline_space_from_configspace(
    configspace: CS.ConfigurationSpace,
) -> dict[str, Parameter]:
    pipeline_space = dict()
    parameter: Parameter
    for hyperparameter in configspace.get_hyperparameters():
        if isinstance(hyperparameter, CS.Constant):
            parameter = ConstantParameter(value=hyperparameter.value)
        elif isinstance(hyperparameter, CS.CategoricalHyperparameter):
            parameter = CategoricalParameter(
                hyperparameter.choices,
                default=hyperparameter.default_value,
            )
        elif isinstance(hyperparameter, CS.OrdinalHyperparameter):
            parameter = CategoricalParameter(
                hyperparameter.sequence,
                default=hyperparameter.default_value,
            )
        elif isinstance(hyperparameter, CS.UniformIntegerHyperparameter):
            parameter = IntegerParameter(
                lower=hyperparameter.lower,
                upper=hyperparameter.upper,
                log=hyperparameter.log,
                default=hyperparameter.default_value,
            )
        elif isinstance(hyperparameter, CS.UniformFloatHyperparameter):
            parameter = FloatParameter(
                lower=hyperparameter.lower,
                upper=hyperparameter.upper,
                log=hyperparameter.log,
                default=hyperparameter.default_value,
            )
        else:
            raise ValueError(f"Unknown hyperparameter type {hyperparameter}")
        pipeline_space[hyperparameter.name] = parameter
    return pipeline_space


class SearchSpace(collections.abc.Mapping):
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

    def compute_prior(self, log: bool = False, ignore_fidelity=False):
        density_value = 0.0 if log else 1.0
        for hyperparameter in self.hyperparameters.values():
            if ignore_fidelity and hyperparameter.is_fidelity:
                continue
            if hasattr(hyperparameter, "has_prior") and hyperparameter.has_prior:
                if log:
                    density_value += hyperparameter.compute_prior(log=True)
                else:
                    density_value *= hyperparameter.compute_prior(log=False)
        return density_value

    def sample(
        self, user_priors: bool = False, patience: int = 1, ignore_fidelity=True
    ) -> SearchSpace:
        sample = self.copy()
        for hp_name, hyperparameter in sample.hyperparameters.items():
            if hyperparameter.is_fidelity and ignore_fidelity:
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
        parent=None,  # pylint: disable=unused-argument
        mutation_rate: float = 1.0,
        mutation_strategy="smbo",
        patience=50,
        **kwargs,
    ):
        if mutation_strategy == "smbo":
            args = {
                "parent": parent,
                "mutation_rate": mutation_rate,
                "mutation_strategy": "local_search",  # fixing property for SMBO mutation
                "patience": patience,
            }
            kwargs.update(args)
            new_config = self._smbo_mutation(**kwargs)
        else:
            raise NotImplementedError("No such mutation strategy!")

        child = SearchSpace(**new_config)

        return child

    def _smbo_mutation(self, patience=50, **kwargs):
        new_config = deepcopy(self.hyperparameters)
        config_hp_names = list(new_config)

        for _ in range(patience):
            idx = random.randint(0, len(new_config) - 1)
            hp_name = config_hp_names[idx]
            hp = new_config[hp_name]
            if isinstance(hp, NumericalParameter) and hp.is_fidelity:
                continue
            try:
                new_config[hp_name] = hp.mutate(**kwargs)
                break
            except Exception as e:
                self.logger.warning(f"{hp_name} FAILED!")
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

    def get_normalized_hp_categories(self, ignore_fidelity=False):
        hps = {
            "continuous": [],
            "categorical": [],
            "graphs": [],
        }
        for hp in self.values():
            hp_value = hp.normalized().value
            if ignore_fidelity and hp.is_fidelity:
                continue
            if isinstance(hp, ConstantParameter):
                continue
            elif isinstance(hp, CategoricalParameter):
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
            if isinstance(hp, ConstantParameter):
                continue
            elif isinstance(hp, CategoricalParameter):
                features["categorical"] += 1
            elif isinstance(hp, NumericalParameter):
                features["continuous"] += 1
        return features

    def set_to_max_fidelity(self):
        self.fidelity.value = self.fidelity.upper

    def get_search_space_grid(self, grid_step_size: int = 10):
        param_ranges = []
        for hp in self.hyperparameters.values():
            if isinstance(hp, Graph):
                raise ValueError("Trying to create a grid for graphs!")
            if isinstance(hp, CategoricalParameter):
                param_ranges.append(hp.choices)
            else:
                if hp.log:
                    grid = np.exp(
                        np.linspace(np.log(hp.lower), np.log(hp.upper), grid_step_size)
                    )
                else:
                    grid = np.linspace(hp.lower, hp.upper, grid_step_size)
                grid = np.clip(grid, hp.lower, hp.upper).astype(np.float32)
                grid = grid.astype(int) if isinstance(hp, IntegerParameter) else grid
                grid = np.unique(grid).tolist()
                param_ranges.append(grid)
        full_grid = product(*param_ranges)

        configs = []
        for _config_dict in full_grid:
            _config = self.copy()
            for key, value in dict(
                zip(self.hyperparameters.keys(), _config_dict)
            ).items():
                _config.add_constant_hyperparameter(name=key, value=value)
            configs.append(_config)
        return configs

    def serialize(self):
        return {key: hp.serialize() for key, hp in self.hyperparameters.items()}

    def load_from(self, config: dict):
        for name in config.keys():
            self.hyperparameters[name].load_from(config[name])

    def copy(self):
        return deepcopy(self)

    def sample_default_configuration(self, patience: int = 1, ignore_fidelity=True):
        config = self.sample(patience=patience, ignore_fidelity=ignore_fidelity)
        for hp_name, hp in self.hyperparameters.items():
            if hp.is_fidelity or isinstance(hp, ConstantParameter):
                continue
            elif hasattr(hp, "default") and hp.default is not None:
                config[hp_name].value = hp.default
            else:
                raise ValueError(f"No defaults specified for {hp} in the space.")
        return config

    def set_defaults_to_current_values(self):
        for hp in self.hyperparameters.values():
            if hasattr(hp, "default"):
                hp.default = hp.value

    def set_hyperparameters_from_dict(
        self,
        hyperparameters,
        defaults=True,
        values=True,
        confidence="low",
        delete_previous_defaults=False,
        delete_previous_values=False
        # If new values / defaults are given, previous defaults / values are always
        # overridden
    ):
        for hp_key, hp in self.hyperparameters.items():
            # First check if there is a new value for the hp and that its value is valid
            if delete_previous_defaults:
                hp.default = None
                hp.has_prior = False
            if delete_previous_values:
                hp.value = None
            if hp_key not in hyperparameters:
                continue
            if self.hyperparameters[hp_key].is_fidelity:
                hp.value = hyperparameters[hp_key]
                continue
            new_hp_value = hyperparameters[hp_key]
            if (
                isinstance(hp, NumericalParameter)
                and not hp.lower <= new_hp_value <= hp.upper
            ):
                continue
            if isinstance(hp, CategoricalParameter) and new_hp_value not in hp.choices:
                continue
            if defaults and hasattr(hp, "default"):
                hp.default = new_hp_value
                self.has_prior = True
                if hasattr(hp, "set_default_confidence_score"):
                    hp.set_default_confidence_score(confidence)
                    print("set confidence to: ", confidence)
            if values:
                hp.value = new_hp_value

    def __getitem__(self, key):
        return self.hyperparameters[key]

    def __iter__(self):
        return iter(self.hyperparameters)

    def __len__(self):
        return len(self.hyperparameters)

    def __str__(self):
        return pprint.pformat(self.hyperparameters)

    def is_equal_value(self, other, include_fidelity=True, on_decimal=8):
        # This does NOT check that the entire SearchSpace is equal (and thus it is not a dunder method),
        # but only checks the configuration
        if self.hyperparameters.keys() != other.hyperparameters.keys():
            return False

        equal_values = [True] * len(self.hyperparameters)
        for hp_idx, hp in enumerate(self.hyperparameters.keys()):
            if self.hyperparameters[hp].is_fidelity and (not include_fidelity):
                continue
            else:
                equal_values[hp_idx] = (
                    self.hyperparameters[hp] == other.hyperparameters[hp]
                )

            if isinstance(self.hyperparameters[hp].value, float):
                equal_values[hp_idx] = (
                    np.round(
                        self.hyperparameters[hp].normalized().value
                        - other.hyperparameters[hp].normalized().value,
                        on_decimal,
                    )
                    == 0
                )
        return all(equal_values)
