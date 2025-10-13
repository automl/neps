"""Validation of the user inputs for NEPS APIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from neps.exceptions import TrialValidationError

if TYPE_CHECKING:
    from neps.space import SearchSpace
    from neps.state.pipeline_eval import UserResultDict


def _validate_imported_result(result: UserResultDict) -> None:
    """Validate the result dictionary e.g., check if objective_to_minimize is present."""
    if result.get("objective_to_minimize") is None:
        raise TrialValidationError(config=result, message="Missing objective_to_minimize")


def _validate_imported_config(
    space: SearchSpace, config: Mapping[str, float]
) -> None | Exception:
    """Validate a configuration against the search space.

    Args:
        space (SearchSpace): The search space to validate against.
        config (dict): The configuration to validate.

    Raises:
        ValueError: If the configuration is not valid.

    """
    all_params = {**space.searchables, **space.fidelities}
    for key in space.searchables:
        if key not in config:
            raise TrialValidationError(config=config, message=f"Missing key: {key}")

    for key, param in all_params.items():
        if key in config and not param.validate(config[key]):
            raise TrialValidationError(
                config=config,
                message=f"Invalid value for parameter: {key}",
            )
    return None
