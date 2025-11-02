"""Validation of the user inputs for NEPS APIs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from neps.space import SearchSpace

if TYPE_CHECKING:
    from neps.space.neps_spaces.parameters import PipelineSpace

logger = logging.getLogger(__name__)


def _normalize_imported_config(  # noqa: C901, PLR0912
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
    else:
        # For PipelineSpace, we need to generate the prefixed keys
        # Import here to avoid circular import
        from neps.space.neps_spaces.neps_space import (
            NepsCompatConverter,
            construct_sampling_path,
        )
        from neps.space.neps_spaces.parameters import Domain

        all_param_keys = set()

        # Add SAMPLING__ prefixed keys for each parameter
        for param_name, param_obj in space.get_attrs().items():
            # Construct the sampling path for this parameter
            if isinstance(param_obj, Domain):
                sampling_path = construct_sampling_path(
                    path_parts=["Resolvable", param_name],
                    domain_obj=param_obj,
                )
                all_param_keys.add(
                    f"{NepsCompatConverter._SAMPLING_PREFIX}{sampling_path}"
                )

        # Add ENVIRONMENT__ prefixed keys for fidelities
        for fidelity_name in space.fidelity_attrs:
            all_param_keys.add(
                f"{NepsCompatConverter._ENVIRONMENT_PREFIX}{fidelity_name}"
            )

    # copy to avoid modifying the original config
    normalized_conf = dict(config)

    fidelities = (
        space.fidelities if isinstance(space, SearchSpace) else space.fidelity_attrs
    )
    for key, param in fidelities.items():
        if key not in normalized_conf:
            normalized_conf[key] = param.upper

    if isinstance(space, SearchSpace):
        extra_keys = set(normalized_conf.keys()) - all_param_keys
    else:
        # For PipelineSpace, filter out keys that match the expected patterns
        # Import here to avoid circular import (needed for prefix constants)
        from neps.space.neps_spaces.neps_space import NepsCompatConverter

        extra_keys = set()
        for key in normalized_conf:
            if not (
                key.startswith(
                    (
                        NepsCompatConverter._SAMPLING_PREFIX,
                        NepsCompatConverter._ENVIRONMENT_PREFIX,
                    )
                )
            ):
                # Check if it's a plain parameter name (without prefix)
                if key not in space.get_attrs() and key not in space.fidelity_attrs:
                    extra_keys.add(key)
            elif key not in all_param_keys:
                # It has a prefix but doesn't match expected sampling/environment keys
                extra_keys.add(key)

    if extra_keys:
        logger.warning(f"Unknown parameters in config: {extra_keys}, discarding them")
        for k in extra_keys:
            normalized_conf.pop(k)
    return normalized_conf
