"""Contains the [`SearchSpace`][neps.search_spaces.search_space.SearchSpace] class
which is a container for hyperparameters that can be sampled, mutated, and crossed over.
"""

from __future__ import annotations

import logging
import operator
import pprint
from collections.abc import Iterator, Mapping
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from neps.search_spaces.parameter import Parameter, ParameterWithPrior
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
        if isinstance(config, str | Path):
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

        if len(config) == 1 and "pipeline_space" in config:
            config = config["pipeline_space"]
        for name, details in config.items():  # type: ignore
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
                        f"Only float and integer fidelities supported, got {hp}"
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

        self.categoricals: Mapping[str, CategoricalParameter] = {
            k: hp for k, hp in _hyperparameters if isinstance(hp, CategoricalParameter)
        }
        self.numerical: Mapping[str, IntegerParameter | FloatParameter] = {
            k: hp
            for k, hp in _hyperparameters
            if isinstance(hp, IntegerParameter | FloatParameter) and not hp.is_fidelity
        }
        self.graphs: Mapping[str, GraphParameter] = {
            k: hp for k, hp in _hyperparameters if isinstance(hp, GraphParameter)
        }
        self.constants: Mapping[str, Any] = {
            k: hp.value for k, hp in _hyperparameters if isinstance(hp, ConstantParameter)
        }
        # NOTE: For future of multiple fidelities
        self.fidelities: Mapping[str, IntegerParameter | FloatParameter] = {}
        if _fidelity_param is not None and _fidelity_name is not None:
            assert isinstance(_fidelity_param, IntegerParameter | FloatParameter)
            self.fidelities = {_fidelity_name: _fidelity_param}

    def set_custom_grid_space(
        self,
        grid_table: pd.Series | pd.DataFrame,
        raw_space: SearchSpace | CS.ConfigurationSpace,
    ) -> None:
        """Set a custom grid space for the search space.

        This function is used to set a custom grid space for the pipeline space.

        !!! warning

            The type check and the table format requirement is loose and
            can break certain components.

        Note:
            Only to be used if a custom set of hyperparameters from the search space
            is to be sampled or used for acquisition functions.
        """
        if grid_table is None or raw_space is None:
            raise ValueError(
                "Both grid_table and raw_space must be set!\n"
                "A table or list of fixed configs must be supported with a "
                "continuous space representing the type and bounds of each "
                "hyperparameter for accurate modeling."
            )

        self.custom_grid_table = grid_table
        self.raw_tabular_space = (
            SearchSpace(**raw_space)
            if not isinstance(raw_space, SearchSpace)
            else raw_space
        )
        self.has_tabular = True

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

            for attempt in range(patience):
                try:
                    if user_priors and isinstance(hp, ParameterWithPrior):
                        sampled_hps[name] = hp.sample(user_priors=user_priors)
                    else:
                        sampled_hps[name] = hp.sample()
                    break
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"Attempt {attempt + 1}/{patience} failed for"
                        f" sampling {name}: {e!s}"
                    )
            else:
                logger.error(
                    f"Failed to sample valid value for {name} after {patience} attempts"
                )
                raise ValueError(
                    f"Could not sample valid value for hyperparameter {name}"
                    f" in {patience} tries!"
                )

        return SearchSpace(**sampled_hps)

    def hp_values(self) -> dict[str, Any]:
        """Get the values for each hyperparameter in this configuration."""
        return {
            hp_name: hp if isinstance(hp, GraphParameter) else hp.value
            for hp_name, hp in self.hyperparameters.items()
        }

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

        !!! note "TODO"

            Include default hyperparameters in the grid.
            If all HPs have a `default` then add a single configuration.
            If only partial HPs have defaults then add all combinations of defaults, but
                only to the end of the list of configs.

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
                    for name, value in zip(
                        self.hyperparameters.keys(), config_values, strict=False
                    )
                }
            )
            for config_values in full_grid
        ]

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
