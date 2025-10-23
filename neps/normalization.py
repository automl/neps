"""Validation of the user inputs for NEPS APIs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neps.space import SearchSpace

logger = logging.getLogger(__name__)


def _normalize_imported_config(space: SearchSpace, config: Mapping[str, float]) -> dict:
    """Completes a configuration by adding default values for missing fidelities.

    Args:
        space: The search space defining the defaults.
        config: The (potentially incomplete) configuration.

    Returns:
        A new, completed configuration dictionary.
    """
    all_param_keys = set(space.searchables.keys()) | set(space.fidelities.keys())

    # copy to avoid modifying the original config
    normalized_conf = dict(config)

    for key, param in space.fidelities.items():
        if key not in normalized_conf:
            normalized_conf[key] = param.upper

    extra_keys = set(normalized_conf.keys()) - all_param_keys
    if extra_keys:
        logger.warning(f"Unknown parameters in config: {extra_keys}, discarding them")
        for k in extra_keys:
            normalized_conf.pop(k)
    return normalized_conf
