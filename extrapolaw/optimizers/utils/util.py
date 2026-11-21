"""Utility functions for trial configuration key mapping and ID assignment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_trial_config_unique_key(
    config: Mapping[str, Any], fid_name: str | None = None
) -> tuple[tuple[str, Any], ...]:
    """
    Generate a unique key for a trial configuration,
    optionally excluding the fidelity parameter.

    Args:
        config: Mapping of configuration parameters.
        fid_name: Name of the fidelity parameter to exclude (if any).

    Returns:
        A tuple of (key, value) pairs sorted by key,
        excluding the fidelity parameter if specified.

    Raises:
        TypeError: If config is not a mapping.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping (e.g., dict).")
    return tuple(
        sorted((k, v) for k, v in config.items() if fid_name is None or k != fid_name)
    )


def _get_max_trial_id(trials: Mapping[str, Any]) -> int | None:
    """Get the maximum numeric trial ID from the trials mapping.
    Args:
        trials: Mapping of trial IDs to Trial objects.
    Returns:
        The maximum numeric trial ID as an integer.
        If no numeric IDs are found, returns 0.
    """
    if not trials:
        return None
    max_id = 0
    for trial_id in trials:
        try:
            # For IDs with a suffix like "1_2", extract the base ID
            parts = trial_id.split("_")
            base_id = int(parts[0])
            max_id = max(max_id, base_id)
        except (ValueError, IndexError):
            # If not numeric, skip
            pass
    return max_id


def _get_next_trial_id(trials: Mapping[str, Any]) -> int:
    """Get the next free numeric trial ID: 0 if there are no trials, else max + 1."""
    max_id = _get_max_trial_id(trials)
    return 0 if max_id is None else max_id + 1
