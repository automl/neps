"""Contains the [`SearchSpace`][neps.search_spaces.search_space.SearchSpace] class
which is a container for hyperparameters that can be sampled, mutated, and crossed over.
"""

from __future__ import annotations

import logging
import operator
import pprint
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Hashable, Iterator, Literal, Mapping
from typing_extensions import Self

import ConfigSpace as CS
import numpy as np
import yaml

from neps.search_spaces.architecture.graph_grammar import GraphParameter
from neps.search_spaces.hyperparameters import (
    CategoricalParameter,
    ConstantParameter,
    FloatParameter,
    IntegerParameter,
    NumericalParameter,
)
from neps.search_spaces.parameter import MutatableParameter, Parameter, ParameterWithPrior
from neps.search_spaces.yaml_search_space_utils import (
    SearchSpaceFromYamlFileError,
    deduce_type,
    formatting_cat,
    formatting_const,
    formatting_float,
    formatting_int,
)
from neps.utils.types import NotSet, _NotSet

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def pipeline_space_from_configspace(
    configspace: CS.ConfigurationSpace,
) -> dict[str, Parameter]:
    """Constructs the [`Parameter`][neps.search_spaces.parameter.Parameter] objects
    from a [`ConfigurationSpace`][ConfigSpace.configuration_space.ConfigurationSpace].

    Args:
        configspace: The configuration space to construct the pipeline space from.

    Returns:
        A dictionary where keys are parameter names and values are parameter objects.
    """
    pipeline_space = {}
    parameter: Parameter
    if any(configspace.get_conditions()) or any(configspace.get_forbiddens()):
        raise NotImplementedError(
            "The ConfigurationSpace has conditions or forbidden clauses, "
            "which are not supported by neps."
        )

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


def pipeline_space_from_yaml(  # noqa: C901
    config: str | Path | dict,
) -> dict[str, Parameter]:
    """Reads configuration details from a YAML file or a dictionary and constructs a
    pipeline space dictionary.

    Args:
        config: Path to the YAML file or a dictionary containing parameter configurations.

    Returns:
        A dictionary where keys are parameter names and values are parameter objects.

    Raises:
        SearchSpaceFromYamlFileError: Raised if there are issues with the YAML file's
            format, contents, or if the dictionary is invalid.
    """
    try:
        if isinstance(config, (str, Path)):
            # try to load the YAML file
            try:
                yaml_file_path = Path(config)
                with yaml_file_path.open("r") as file:
                    config = yaml.safe_load(file)
                if not isinstance(config, dict):
                    raise ValueError(
                        "The loaded pipeline_space is not a valid dictionary. Please "
                        "ensure that you use a proper structure. See the documentation "
                        "for more details."
                    )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Unable to find the specified file for 'pipeline_space' at "
                    f"'{config}'. Please verify the path specified in the "
                    f"'pipeline_space' argument and try again."
                ) from e
            except yaml.YAMLError as e:
                raise ValueError(f"The file at {config} is not a valid YAML file.") from e

        pipeline_space: dict[str, Parameter] = {}

        for name, details in config.items():
            param_type = deduce_type(name, details)

            if param_type in ("int", "integer"):
                formatted_details = formatting_int(name, details)
                pipeline_space[name] = IntegerParameter(**formatted_details)
            elif param_type == "float":
                formatted_details = formatting_float(name, details)
                pipeline_space[name] = FloatParameter(**formatted_details)
            elif param_type in ("cat", "categorical"):
                formatted_details = formatting_cat(name, details)
                pipeline_space[name] = CategoricalParameter(**formatted_details)
            elif param_type == "const":
                const_details = formatting_const(details)
                pipeline_space[name] = ConstantParameter(const_details)
            else:
                # Handle unknown parameter type
                raise TypeError(
                    f"Unsupported parameter with details: {details} for '{name}'.\n"
                    f"Supported Types for argument type are:\n"
                    "For integer parameter: int, integer\n"
                    "For float parameter: float\n"
                    "For categorical parameter: cat, categorical\n"
                    "Constant parameter was not detect\n"
                )
    except (KeyError, TypeError, ValueError, FileNotFoundError) as e:
        raise SearchSpaceFromYamlFileError(e) from e

    return pipeline_space


class SearchSpace(Mapping[str, Any]):
    """A container for hyperparameters that can be sampled, mutated, and crossed over.

    Provides operations for operating on and generating new configurations from the
    hyperparameters.

    !!! note

        The `SearchSpace` class is both the definition of the search space and also
        a configuration at the same time.

        When refering to the `SearchSpace` as a configuration, the documentation will
        refer to it as a `configuration` or `config`. Otherwise, it will be referred to
        as a `search space`.

    !!! note "TODO"

        This documentation is WIP. If you have any questions, please reach out so we can
        know better what to document.
    """

    def __init__(self, **hyperparameters: Parameter):
        """Initialize the SearchSpace with hyperparameters.

        Args:
            **hyperparameters: The hyperparameters that define the search space.
        """
        # Ensure a consistent ordering for uses throughout the lib
        _hyperparameters = sorted(hyperparameters.items(), key=lambda x: x[0])
        _fidelity_param: NumericalParameter | None = None
        _fidelity_name: str | None = None
        _has_prior: bool = False

        for name, hp in _hyperparameters:
            if hp.is_fidelity:
                if _fidelity_param is not None:
                    raise ValueError(
                        "neps only supports one fidelity parameter in the pipeline space,"
                        " but multiple were given. (Hint: check you pipeline space for "
                        "multiple is_fidelity=True)"
                    )

                if not isinstance(hp, NumericalParameter):
                    raise ValueError(
                        "neps only suport float and integer fidelity parameters"
                    )

                _fidelity_param = hp
                _fidelity_name = name

            if isinstance(hp, ParameterWithPrior) and hp.has_prior:
                _has_prior = True

        self.hyperparameters: dict[str, Parameter] = dict(_hyperparameters)
        self.fidelity: NumericalParameter | None = _fidelity_param
        self.fidelity_name: str | None = _fidelity_name
        self.has_prior: bool = _has_prior

        # TODO(eddiebergman): This should be a seperate thing most likely and not
        # in a `SearchSpace`.
        # Variables for tabular bookkeeping
        self.custom_grid_table: pd.Series | pd.DataFrame | None = None
        self.raw_tabular_space: SearchSpace | None = None
        self.has_tabular: bool = False

    @property
    def has_fidelity(self) -> bool:
        """Check if the search space has a fidelity parameter."""
        return self.fidelity is not None

    def compute_prior(self, *, log: bool = False, ignore_fidelity: bool = False) -> float:
        """Compute the prior probability of the search space.

        This is better know as the `pdf` of the configuraiton in the search space, or a
        relative measure of how likely this configuration is under the search space.

        Args:
            log: Whether to compute the log of the prior.
            ignore_fidelity: Whether to ignore the fidelity parameter when
                computing the prior.


        Returns:
            The likelihood of the configuration in the search space.
        """
        density_value = 0.0 if log else 1.0
        op = operator.add if log else operator.mul

        prior_hps = (
            hp
            for hp in self.hyperparameters.values()
            if isinstance(hp, ParameterWithPrior) and hp.has_prior
        )

        for hyperparameter in prior_hps:
            if ignore_fidelity and hyperparameter.is_fidelity:
                continue

            hp_prior = hyperparameter.compute_prior(log=log)
            density_value = op(density_value, hp_prior)

        return density_value

    def sample(
        self,
        *,
        user_priors: bool = False,
        patience: int = 1,
        ignore_fidelity: bool = True,
    ) -> SearchSpace:
        """Sample a configuration from the search space.

        Args:
            user_priors: Whether to use user priors when sampling.
            patience: The number of times to try to sample a valid value for a
                hyperparameter.
            ignore_fidelity: Whether to ignore the fidelity parameter when sampling.

        Returns:
            A sampled configuration from the search space.
        """
        sampled_hps: dict[str, Parameter] = {}

        for name, hp in self.hyperparameters.items():
            if hp.is_fidelity and ignore_fidelity:
                sampled_hps[name] = hp.clone()
                continue

            for _ in range(patience):
                try:
                    if user_priors and isinstance(hp, ParameterWithPrior):
                        sampled_hps[name] = hp.sample(user_priors=user_priors)
                    else:
                        sampled_hps[name] = hp.sample()
                    break
                except ValueError:
                    logger.warning(
                        f"Could not sample valid value for hyperparameter {name}!"
                    )
            else:
                raise ValueError(
                    f"Could not sample valid value for hyperparameter {name}"
                    f" in {patience} tries!"
                )

        return SearchSpace(**sampled_hps)

    def mutate(
        self,
        *,
        parent: SearchSpace | None = None,
        mutation_rate: float = 1.0,
        mutation_strategy: Literal["smbo"] = "smbo",
        patience: int = 50,
        **kwargs: Any,
    ) -> SearchSpace:
        """Mutate the search space.

        Args:
            parent: The parent configuration to mutate from.
            mutation_rate: The rate at which to mutate the search space.
            mutation_strategy: The strategy to use for mutation.
            patience: The number of times to try to mutate a valid value for a
                hyperparameter.
            **kwargs: Additional keyword arguments to pass to the mutation strategy.

        Returns:
            The mutated search space.
        """
        if mutation_strategy == "smbo":
            args = {
                "parent": parent,
                "mutation_rate": mutation_rate,
                "mutation_strategy": "local_search",  # fixing property for SMBO mutation
            }
            kwargs.update(args)
            new_config = self._smbo_mutation(patience=patience, **kwargs)
        else:
            raise NotImplementedError("No such mutation strategy!")

        return SearchSpace(**new_config)

    # TODO(eddiebergman): This function seems very weak, i.e. it's only mutating
    # one hyperparamter and copying the rest, very expensive for little gain.
    def _smbo_mutation(self, *, patience: int = 5, **kwargs: Any) -> Self:
        non_fidelity_mutatable_params = {
            hp_name: hp
            for hp_name, hp in self.hyperparameters.items()
            if not hp.is_fidelity and isinstance(hp, MutatableParameter)
        }

        for _ in range(patience):
            chosen_hp_name = np.random.choice(list(non_fidelity_mutatable_params))
            hp = non_fidelity_mutatable_params[chosen_hp_name]

            try:
                mutated_param = hp.mutate(**kwargs)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{chosen_hp_name} failed to mutate! Error: {e}, {kwargs}")
                continue

            new_params = {
                hp_name: hp.clone() if hp_name != chosen_hp_name else mutated_param
                for hp_name, hp in self.hyperparameters.items()
            }
            return self.__class__(**new_params)

        raise ValueError(
            f"Could not mutate valid value for hyperparameter in {patience} tries!"
        )

    def crossover(
        self,
        config2: SearchSpace,
        crossover_probability_per_hyperparameter: float = 1.0,
        patience: int = 50,
        crossover_strategy: str = "simple",
    ) -> tuple[SearchSpace, SearchSpace]:
        """Crossover this configuration with another.

        Args:
            config2: The other search space to crossover with.
            crossover_probability_per_hyperparameter: The probability of crossing over
                each hyperparameter.
            patience: The number of times to try to crossover a valid value for a
                hyperparameter.
            crossover_strategy: The strategy to use for crossover.

        Returns:
            A tuple of the two new configurations.
        """
        if crossover_strategy == "simple":
            new_config1, new_config2 = self._simple_crossover(
                config2=config2,
                crossover_probability_per_hyperparameter=crossover_probability_per_hyperparameter,
                patience=patience,
            )
        else:
            raise NotImplementedError("No such crossover strategy!")

        if len(self.hyperparameters.keys()) != len(new_config1):
            raise Exception("Cannot crossover")

        return SearchSpace(**new_config1), SearchSpace(**new_config2)

    def _simple_crossover(
        self,
        config2: SearchSpace,
        crossover_probability_per_hyperparameter: float = 1.0,
        patience: int = 50,
    ) -> tuple[dict[str, Parameter], dict[str, Parameter]]:
        new_config1: dict[str, Parameter] = {}
        new_config2: dict[str, Parameter] = {}

        for key, hyperparameter in self.hyperparameters.items():
            other_hp = config2.hyperparameters[key]
            if (
                isinstance(hyperparameter, MutatableParameter)
                and not hyperparameter.is_fidelity
                and np.random.random() < crossover_probability_per_hyperparameter
            ):
                for _ in range(patience):
                    try:
                        child1, child2 = hyperparameter.crossover(other_hp)  # type: ignore
                        new_config1[key] = child1
                        new_config2[key] = child2
                    except Exception:  # noqa: S112, BLE001
                        continue
                    else:
                        break
            else:
                new_config1[key] = hyperparameter.clone()
                new_config2[key] = other_hp.clone()

        return new_config1, new_config2

    def get_normalized_hp_categories(
        self,
        *,
        ignore_fidelity: bool = False,
    ) -> dict[Literal["continuous", "categorical", "graphs"], list[Any]]:
        """Get the normalized values for each hyperparameter in the configuraiton.

        Args:
            ignore_fidelity: Whether to ignore the fidelity parameter when getting the
                normalized values.

        Returns:
            A dictionary of the normalized values for each hyperparameter,
            separated by type.
        """
        hps: dict[Literal["continuous", "categorical", "graphs"], list[Any]] = {
            "continuous": [],
            "categorical": [],
            "graphs": [],
        }
        for hp in self.values():
            if ignore_fidelity and hp.is_fidelity:
                continue

            if isinstance(hp, ConstantParameter):
                continue

            # TODO(eddiebergman): Not sure this covers all graph parameters but a search
            # for `def value` that have a property decorator is all that could have
            # worked previously for graphs
            if isinstance(hp, GraphParameter):
                hps["graphs"].append(hp.value)

            elif isinstance(hp, CategoricalParameter):
                assert hp.value is not None
                hp_value = hp.value_to_normalized(hp.value)
                hps["categorical"].append(hp_value)

            # TODO(eddiebergman): Technically integer is not continuous
            elif isinstance(hp, NumericalParameter):
                assert hp.value is not None
                hp_value = hp.value_to_normalized(hp.value)
                hps["continuous"].append(hp_value)
            else:
                raise NotImplementedError(f"Unknown Parameter type: {type(hp)}\n{hp}")

        return hps

    def hp_values(self) -> dict[str, Any]:
        """Get the values for each hyperparameter in this configuration."""
        return {
            hp_name: hp if isinstance(hp, GraphParameter) else hp.value
            for hp_name, hp in self.hyperparameters.items()
        }

    def add_hyperparameter(self, name: str, hp: Parameter) -> None:
        """Add a hyperparameter to the search space.

        Args:
            name: The name of the hyperparameter.
            hp: The hyperparameter to add.
        """
        self.hyperparameters[str(name)] = hp
        self.hyperparameters = dict(
            sorted(self.hyperparameters.items(), key=lambda x: x[0])
        )

    def get_vectorial_dim(self) -> dict[Literal["continuous", "categorical"], int] | None:
        """Get the vectorial dimension of the search space.

        The count of [`NumericalParameter`][neps.search_spaces.NumericalParameter]
        are put under the key `#!python "continuous"` and the count of
        [`CategoricalParameter`][neps.search_spaces.CategoricalParameter] are put under
        the key `#!python "categorical"` in the return dict.

        If there are no numerical or categorical hyperparameters **or constant**
        parameters, then `None` is returned.

        Returns:
            The vectorial dimension
        """
        if not any(
            isinstance(hp, (NumericalParameter, CategoricalParameter, ConstantParameter))
            for hp in self.values()
        ):
            return None

        features: dict[Literal["continuous", "categorical"], int] = {
            "continuous": 0,
            "categorical": 0,
        }
        for hp in self.values():
            if isinstance(hp, ConstantParameter):
                pass
            elif isinstance(hp, GraphParameter):
                # TODO(eddiebergman): This was what the old behaviour would do...
                pass
            elif isinstance(hp, CategoricalParameter):
                features["categorical"] += 1
            elif isinstance(hp, NumericalParameter):
                features["continuous"] += 1
            else:
                raise NotImplementedError(f"Unknown Parameter type: {type(hp)}\n{hp}")

        return features

    def set_to_max_fidelity(self) -> None:
        """Set the configuration to the maximum fidelity."""
        if self.fidelity is None:
            raise ValueError("No fidelity parameter in the search space!")

        self.fidelity.set_value(self.fidelity.upper)

    def get_search_space_grid(
        self,
        *,
        size_per_numerical_hp: int = 10,
        include_endpoints: bool = True,
    ) -> list[SearchSpace]:
        """Get a grid of configurations from the search space.

        For [`NumericalParameter`][neps.search_spaces.NumericalParameter] hyperparameters,
        the parameter `size_per_numerical_hp=` is used to determine a grid. If there are
        any duplicates, e.g. for an
        [`IntegerParameter`][neps.search_spaces.IntegerParameter], then we will
        remove duplicates.

        For [`CategoricalParameter`][neps.search_spaces.CategoricalParameter]
        hyperparameters, we include all the choices in the grid.

        For [`ConstantParameter`][neps.search_spaces.ConstantParameter] hyperparameters,
        we include the constant value in the grid.

        !!! note "TODO"

            Does not support graph parameters currently.

        Args:
            size_per_numerical_hp: The size of the grid for each numerical hyperparameter.
            include_endpoints: Whether to include the endpoints of the grid.

        Returns:
            A list of configurations from the search space.
        """
        param_ranges = []
        for hp in self.hyperparameters.values():
            # NOTE(eddiebergman): This is a temporary fix to avoid graphs
            # If this is resolved, please update the docstring!
            if isinstance(hp, GraphParameter):
                raise ValueError("Trying to create a grid for graphs!")

            if isinstance(hp, CategoricalParameter):
                param_ranges.append(hp.choices)
                continue

            if isinstance(hp, ConstantParameter):
                param_ranges.append([hp.value])
                continue

            if isinstance(hp, NumericalParameter):
                grid = hp.grid(
                    size=size_per_numerical_hp,
                    include_endpoint=include_endpoints,
                )
                _grid = np.clip(grid, hp.lower, hp.upper).astype(np.float64)
                _grid = (
                    _grid.astype(np.int64) if isinstance(hp, IntegerParameter) else _grid
                )
                _grid = np.unique(grid).tolist()
                param_ranges.append(grid)
                continue

            raise NotImplementedError(f"Unknown Parameter type: {type(hp)}\n{hp}")

        full_grid = product(*param_ranges)

        return [
            SearchSpace(
                **{
                    name: ConstantParameter(value=value)  # type: ignore
                    for name, value in zip(self.hyperparameters.keys(), config_values)
                }
            )
            for config_values in full_grid
        ]

    def serialize(self) -> dict[str, Hashable]:
        """Serialize the configuration to a dictionary that can be written to disk."""
        serialized_config = {}
        for name, hp in self.hyperparameters.items():
            if hp.value is None:
                raise ValueError(
                    f"Hyperparameter {name} has no value set and can't" " be serialized!"
                )
            serialized_config[name] = hp.serialize_value(hp.value)
        return serialized_config

    def from_dict(self, config: Mapping[str, Any | GraphParameter]) -> SearchSpace:
        """Create a new instance of this search space with parameters set from the config.

        Args:
            config: The dictionary of hyperparameters to set with values.
        """
        new = self.clone()
        for name, val in config.items():
            new.hyperparameters[name].load_from(val)

        return new

    def clone(self, *, _with_tabular: bool = False) -> SearchSpace:
        """Create a copy of the search space."""
        new_copy = self.__class__(
            **{k: v.clone() for k, v in self.hyperparameters.items()}
        )
        if _with_tabular and self.has_tabular:
            assert self.custom_grid_table is not None
            assert self.raw_tabular_space is not None
            new_copy.set_custom_grid_space(
                grid_table=self.custom_grid_table,
                raw_space=self.raw_tabular_space,
            )

        return new_copy

    def sample_default_configuration(
        self,
        *,
        patience: int = 1,
        ignore_fidelity: bool = True,
        ignore_missing_defaults: bool = False,
    ) -> SearchSpace:
        """Sample the default configuration from the search space.

        By default, if there is no default set for a hyperparameter, an error will be
        raised. If `ignore_missing_defaults=True`, then a sampled value will be used
        instead.

        Args:
            patience: The number of times to try to sample a valid value for a
                hyperparameter.
            ignore_fidelity: Whether to ignore the fidelity parameter when sampling.
            ignore_missing_defaults: Whether to ignore missing defaults when setting
                the default configuration.

        Returns:
            The default configuration.
        """
        # Sample a random config and then set the defaults if there are any
        config = self.sample(patience=patience, ignore_fidelity=ignore_fidelity)
        for hp_name, hp in self.hyperparameters.items():
            if hp.is_fidelity and ignore_fidelity:
                continue

            if hp.default is None:
                if not ignore_missing_defaults:
                    raise ValueError(f"No defaults specified for {hp} in the space.")

                # Use the sampled value instead
            else:
                config[hp_name].set_value(hp.default)

        return config

    def set_defaults_to_current_values(self) -> None:
        """Update the configuration/search space to use the current values as defaults."""
        for hp in self.hyperparameters.values():
            if isinstance(hp, NumericalParameter):
                hp.set_default(hp.value)

    def set_hyperparameters_from_dict(  # noqa: C901
        self,
        hyperparameters: Mapping[str, Any],
        *,
        defaults: bool = True,
        values: bool = True,
        # TODO(eddiebergman): The existence of this makes me think
        # all hyperparameters that accept confidence should use the same keys
        confidence: str = "low",
        delete_previous_defaults: bool = False,
        delete_previous_values: bool = False,
        overwrite_constants: bool = False,
    ) -> None:
        """Set the hyperparameters from a dictionary of values.

        !!! note "Constant Hyperparameters"

            [`ConstantParameter`][neps.search_spaces.ConstantParameter] hyperparameters
            have only a single possible value and hence only a single possible default.
            If `overwrite_constants=` is `False`, then it will remain unchanged and
            ignore the new value.

            If `overwrite_constants=` is `True`, then the constant hyperparameter will
            be updated, requiring both `defaults=True` and `values=True` to be set.

            The arguments `delete_previous_defaults` and `delete_previous_values` are
            ignored for [`ConstantParameter`][neps.search_spaces.ConstantParameter].

        Args:
            hyperparameters: The dictionary of hyperparameters to set with values.
            defaults: Whether to set the defaults to these values.
            values: Whether to set the value of the hyperparameters to these values.
            confidence: The confidence score to use when setting the default.
                Only applies if `defaults=True`.
            delete_previous_defaults: Whether to delete the previous defaults.
            delete_previous_values: Whether to delete the previous values.
            overwrite_constants: Whether to overwrite constant hyperparameters.

        Raises:
            ValueError: If the value is invalid for the hyperparameter.
        """
        if values is False and defaults is False:
            raise ValueError("At least one of `values` or `defaults` must be True.")

        for hp_key, current_hp in self.hyperparameters.items():
            new_hp_value = hyperparameters.get(hp_key, NotSet)
            if isinstance(new_hp_value, _NotSet):
                continue

            # Handle constants specially as they have particular logic which
            # is different from the other hyperparameters
            if isinstance(current_hp, ConstantParameter):
                if not overwrite_constants:
                    continue

                if not (defaults and values):
                    raise ValueError(
                        "Cannot have a constant parameter with a seperate default and"
                        " and value. Please provide both `values=True` and"
                        " `defaults=True` if passing `overwrite_constants=True`"
                        f" with a new value for the constant '{hp_key}'."
                    )

                current_hp.set_constant_value(new_hp_value)
                continue

            if delete_previous_defaults:
                current_hp.set_default(None)

            if delete_previous_values:
                current_hp.set_value(None)

            if defaults:
                current_hp.set_default(new_hp_value)
                if isinstance(current_hp, ParameterWithPrior):
                    current_hp.set_default_confidence_score(confidence)

            if values:
                current_hp.set_value(new_hp_value)

    def __getitem__(self, key: str) -> Parameter:
        return self.hyperparameters[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.hyperparameters)

    def __len__(self) -> int:
        return len(self.hyperparameters)

    def __str__(self) -> str:
        return pprint.pformat(self.hyperparameters)

    def is_equal_value(
        self,
        other: SearchSpace,
        *,
        include_fidelity: bool = True,
        on_decimal: int = 8,
    ) -> bool:
        """Check if the configuration is equal to another configuration.

        !!! warning

            This does **NOT** check that the entire `SearchSpace` is equal (and thus it is
            not a dunder method), but only checks the configuration values.

        Args:
            other: The other configuration to compare to.
            include_fidelity: Whether to include the fidelity parameter in the comparison.
            on_decimal: The decimal to round to when comparing float values.

        Returns:
            Whether the configuration values are equal.
        """
        if self.hyperparameters.keys() != other.hyperparameters.keys():
            return False

        for hp_key, this_hp in self.hyperparameters.items():
            if this_hp.is_fidelity and (not include_fidelity):
                continue

            other_hp = other.hyperparameters[hp_key]
            if not isinstance(other_hp, type(this_hp)):
                return False

            if isinstance(this_hp.value, float):
                this_norm = this_hp.value_to_normalized(this_hp.value)
                other_norm = other_hp.value_to_normalized(other_hp.value)  # type: ignore
                if np.round(this_norm - other_norm, on_decimal) != 0:
                    return False
            elif this_hp.value != other_hp.value:
                return False

        return True
