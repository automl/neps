"""Validation of the user inputs for NEPS APIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from typing_extensions import assert_never

from neps.exceptions import TrialValidationError
from neps.space import SearchSpace
from neps.space.neps_spaces.parameters import PipelineSpace

if TYPE_CHECKING:
    from neps.space.neps_spaces.parameters import Categorical, Float, Integer
    from neps.space.parameters import HPOCategorical, HPOConstant, HPOFloat, HPOInteger
    from neps.state.pipeline_eval import UserResultDict


def _validate_imported_result(result: UserResultDict) -> None:
    """Validate the result dictionary e.g., check if objective_to_minimize is present."""
    if result.get("objective_to_minimize") is None:
        raise TrialValidationError(config=result, message="Missing objective_to_minimize")


def validate_parameter_value(
    param: (
        HPOFloat
        | HPOInteger
        | HPOCategorical
        | HPOConstant
        | Float
        | Integer
        | Categorical
    ),
    value: Any,
) -> bool:
    """Validate a parameter value against its parameter definition.

    Works with both SearchSpace parameters (HPOFloat, HPOInteger, HPOCategorical,
    HPOConstant) and PipelineSpace parameters (Float, Integer, Categorical from
    neps_spaces).

    Args:
        param: The parameter definition (from either SearchSpace or PipelineSpace)
        value: The value to validate

    Returns:
        bool: True if the value is valid for the parameter, False otherwise
    """
    # Import here to avoid circular dependencies
    from neps.space.neps_spaces.parameters import (
        Categorical as NepsCategorical,
        Float as NepsFloat,
        Integer as NepsInteger,
    )
    from neps.space.parameters import (
        HPOCategorical,
        HPOConstant,
        HPOFloat,
        HPOInteger,
    )

    # Float parameters - both use .lower and .upper
    if isinstance(param, HPOFloat | NepsFloat):
        return isinstance(value, float | int) and param.lower <= value <= param.upper

    # Integer parameters - both use .lower and .upper
    if isinstance(param, HPOInteger | NepsInteger):
        return isinstance(value, int) and param.lower <= value <= param.upper

    # Categorical parameters - both use .choices
    if isinstance(param, HPOCategorical):
        choices = param.choices
        return value in choices
    if isinstance(param, NepsCategorical):
        return (
            0
            <= value
            < (len(list(param.choices)) if isinstance(param.choices, tuple) else 1)
        )

    # Constant - SearchSpace only
    if isinstance(param, HPOConstant):
        return value == param.value

    # Exhaustiveness check - all cases should be covered
    assert_never(param)


def _validate_imported_config(  # noqa: C901, PLR0912
    space: SearchSpace | PipelineSpace, config: Mapping[str, float]
) -> None:
    """Validate a configuration against the search space.

    Args:
        space: The search space to validate against.
        config: The configuration to validate.

    Raises:
        TrialValidationError: If the configuration is not valid.

    """
    if isinstance(space, SearchSpace):
        all_params = {**space.searchables, **space.fidelities}
        for key in space.searchables:
            if key not in config:
                raise TrialValidationError(config=config, message=f"Missing key: {key}")

        for key, param in all_params.items():
            if key in config and not validate_parameter_value(param, config[key]):
                raise TrialValidationError(
                    config=config,
                    message=f"Invalid value for parameter: {key}",
                )
    elif isinstance(space, PipelineSpace):
        # For PipelineSpace, we need to check for the prefixed keys
        # Import here to avoid circular import
        from neps.space.neps_spaces.neps_space import (
            NepsCompatConverter,
            construct_sampling_path,
        )
        from neps.space.neps_spaces.parameters import Domain

        # Check that all expected parameter keys are present in the config
        for param_name, param_obj in space.get_attrs().items():
            if isinstance(param_obj, Domain):
                # Construct the expected sampling path
                sampling_path = construct_sampling_path(
                    path_parts=["Resolvable", param_name],
                    domain_obj=param_obj,
                )
                expected_key = f"{NepsCompatConverter._SAMPLING_PREFIX}{sampling_path}"
                if expected_key not in config:
                    raise TrialValidationError(
                        config=config, message=f"Missing key: {expected_key}"
                    )

        # Check that all expected fidelity keys are present in the config
        for fidelity_name in space.fidelity_attrs:
            expected_key = f"{NepsCompatConverter._ENVIRONMENT_PREFIX}{fidelity_name}"
            if expected_key not in config:
                raise TrialValidationError(
                    config=config, message=f"Missing fidelity key: {expected_key}"
                )

        # Validate parameter values for PipelineSpace
        # Note: PipelineSpace doesn't have a searchables attribute like SearchSpace
        # We need to validate the attrs that are actual parameters
        from neps.space.neps_spaces.parameters import Categorical, Float, Integer

        for param_name, param in space.get_attrs().items():  # type: ignore[unreachable]
            if isinstance(param, Float | Integer | Categorical):  # type: ignore[unreachable]
                # Construct the expected sampling path and key
                sampling_path = construct_sampling_path(  # type: ignore[unreachable]
                    path_parts=["Resolvable", param_name],
                    domain_obj=param,
                )
                expected_key = f"{NepsCompatConverter._SAMPLING_PREFIX}{sampling_path}"

                # Validate the value if the key is present in config
                if expected_key in config and not validate_parameter_value(
                    param, config[expected_key]
                ):
                    raise TrialValidationError(
                        config=config,
                        message=f"Invalid value for parameter: {expected_key}",
                    )
