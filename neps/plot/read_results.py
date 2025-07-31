"""Utility functions for reading and processing results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import neps
from neps.state.trial import State


def process_seed(
    *,
    path: str | Path,
    seed: str | int | None,
    key_to_extract: str | None = None,
    consider_continuations: bool = False,
    n_workers: int = 1,
) -> tuple[list[float], list[float], float]:
    """Reads and processes data per seed."""
    path = Path(path)
    if seed is not None:
        path = path / str(seed) / "neps_root_directory"

    _fulldf, _summary = neps.status(path, print_summary=False)
    if _fulldf.empty:
        raise ValueError(f"No trials found in {path}")

    _fulldf = _fulldf.sort_values("time_sampled")

    def get_cost(idx: str | int) -> float:
        row = _fulldf.loc[idx]
        if key_to_extract and key_to_extract in row:
            return float(row[key_to_extract])
        return 1.0

    losses = []
    costs = []

    # max_cost only relevant for scaling x-axis when using fidelity on the x-axis
    max_cost: float = -1.0
    global_start = _fulldf["time_sampled"].min()

    for config_id, config_result in _fulldf.iterrows():
        if config_result["state"] != State.SUCCESS:
            continue

        cost = get_cost(config_id)

        if (
            consider_continuations
            and n_workers == 1
            and "previous_config_id" in config_result["metadata"]
        ):
            # calculates continuation costs for MF algorithms NOTE: assumes that
            # all recorded evaluations are black-box evaluations where
            # continuations or freeze-thaw was not accounted for during optimization
            previous_id = config_result["metadata"]["previous_config_id"]

            if previous_id in _fulldf.index and key_to_extract:
                cost -= get_cost(config_id)
            else:
                cost = float(config_result["time_end"] - global_start)

        loss = float(config_result["objective_to_minimize"])
        losses.append(loss)
        costs.append(cost)
        max_cost = max(max_cost, cost)

    return list(np.minimum.accumulate(losses)), costs, max_cost
