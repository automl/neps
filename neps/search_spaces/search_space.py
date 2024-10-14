"""Contains the [`SearchSpace`][neps.search_spaces.search_space.SearchSpace] class
which is a container for hyperparameters that can be sampled, mutated, and crossed over.
"""

from __future__ import annotations

import logging
import pprint
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import ConfigSpace as CS
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
                pipeline_space[name] = ConstantParameter(const_details)  # type: ignore
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

    def from_dict(self, config: Mapping[str, Any | GraphParameter]) -> SearchSpace:
        """Create a new instance of this search space with parameters set from the config.

        Args:
            config: The dictionary of hyperparameters to set with values.
        """
        new = self.clone()
        for name, val in config.items():
            new.hyperparameters[name].load_from(val)

        return new

    def clone(self) -> SearchSpace:
        """Create a copy of the search space."""
        return self.__class__(**{k: v.clone() for k, v in self.hyperparameters.items()})

    def sample_default_configuration(
        self,
        *,
        patience: int = 1,
        ignore_fidelity: bool = True,
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
                raise ValueError(f"No defaults specified for {hp} in the space.")

                # Use the sampled value instead
            config[hp_name].set_value(hp.default)

        return config

    def __getitem__(self, key: str) -> Parameter:
        return self.hyperparameters[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.hyperparameters)

    def __len__(self) -> int:
        return len(self.hyperparameters)

    def __str__(self) -> str:
        return pprint.pformat(self.hyperparameters)
