"""Validation of the user inputs for NEPS APIs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from neps.space import SearchSpace

if TYPE_CHECKING:
    from neps.space.neps_spaces.parameters import PipelineSpace

logger = logging.getLogger(__name__)


def _normalize_imported_config(
    space: SearchSpace | PipelineSpace, config: Mapping[str, float]
) -> dict:
    """Completes a configuration by adding default values for missing fidelities.

    Args:
        space: The search space defining the defaults.
        config: The (potentially incomplete) configuration.

    Returns:
        A new, completed configuration dictionary.
    """
    if isinstance(space, SearchSpace):
        all_param_keys = set(space.searchables.keys()) | set(space.fidelities.keys())
        # copy to avoid modifying the original config
        normalized_conf = dict(config)
        for key, param in space.fidelities.items():
            if key not in normalized_conf:
                normalized_conf[key] = param.upper
        extra_keys = set(normalized_conf.keys()) - all_param_keys
    else:
        # For PipelineSpace, we need to generate the prefixed keys
        # Import here to avoid circular import
        from neps.space.neps_spaces.neps_space import (
            NepsCompatConverter,
        )

        # copy to avoid modifying the original config
        normalized_conf = dict(config)

        for key, fid_param in space.fidelity_attrs.items():
            fid_key = NepsCompatConverter._ENVIRONMENT_PREFIX + key
            if fid_key not in normalized_conf:
                normalized_conf[fid_key] = fid_param.upper
        # For PipelineSpace, filter out keys that match the expected patterns
        # Import here to avoid circular import (needed for prefix constants)
        from neps.space.neps_spaces.neps_space import NepsCompatConverter

        extra_keys = set()
        for key in normalized_conf:
            if not key.startswith(
                (
                    NepsCompatConverter._SAMPLING_PREFIX,
                    NepsCompatConverter._ENVIRONMENT_PREFIX,
                )
            ):
                # It has no prefix.
                # TODO: It might still be unnecessary, but it will not hurt.
                extra_keys.add(key)

    if extra_keys:
        logger.warning(f"Unknown parameters in config: {extra_keys}, discarding them")
        for k in extra_keys:
            normalized_conf.pop(k)
    return normalized_conf
