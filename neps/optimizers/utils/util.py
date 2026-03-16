"""Utility functions for trial configuration key mapping and ID assignment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


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


def get_config_key_to_id_mapping(
    table: pd.DataFrame, fid_name: str
) -> dict[tuple[tuple[str, Any], ...], int]:
    """
    Map each unique trial configuration (excluding fidelity) to its highest config ID.

    Args:
        table: DataFrame with trial configurations and IDs.
        fid_name: Name of the fidelity parameter to exclude.

    Returns:
        Dictionary mapping unique config keys to their highest config ID.

    Raises:
        TypeError: If table is not a pandas DataFrame.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame.")
    config_to_id = {}
    for idx, row in table.iterrows():
        config_id = idx[0]
        config = row["config"]
        config_key = get_trial_config_unique_key(config, fid_name)
        # we always want to keep the highest config_id for a given config_key
        config_to_id[config_key] = config_id
    return config_to_id


def _get_max_trial_id(trials: Mapping[str, Any]) -> int:
    """Get the maximum numeric trial ID from the trials mapping.
    Args:
        trials: Mapping of trial IDs to Trial objects.
    Returns:
        The maximum numeric trial ID as an integer.
        If no numeric IDs are found, returns 0.
    """
    if not trials:
        return 0
    max_id = 0
    for trial_id in trials:
        try:
            # For hierarchical IDs like "1_rung_2", extract the base ID
            parts = trial_id.split("_")
            base_id = int(parts[0])
            max_id = max(max_id, base_id)
        except (ValueError, IndexError):
            # If not numeric, skip
            pass
    return max_id
