from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np

import neps


def process_seed(
    path: str | os.PathLike,
    seed: str | int | None,
    key_to_extract: str | None = None,
    consider_continuations: bool = False,
    n_workers: int = 1,
) -> tuple[list[float], list[float], float]:
    """Reads and processes data per seed."""

    if seed is not None:
        path = os.path.join(path, str(seed), "neps_root_directory")

    stats, _ = neps.status(path, print_summary=False)
    sorted_stats = sorted(sorted(stats.items()), key=lambda x: len(x[0]))
    stats = OrderedDict(sorted_stats)

    # max_cost only relevant for scaling x-axis when using fidelity on the x-axis
    max_cost = -1.0
    if key_to_extract == "fidelity":
        max_cost = max(s.result["info_dict"][key_to_extract] for s in stats.values())

    global_start = stats[min(stats.keys())].metadata["time_sampled"]

    def get_cost(idx):
        if key_to_extract is not None:
            return stats[idx].result["info_dict"][key_to_extract]
        return 1.0

    losses = []
    costs = []

    for config_id, config_result in stats.items():
        config_cost = get_cost(config_id)
        if consider_continuations:
            if n_workers == 1:
                # calculates continuation costs for MF algorithms NOTE: assumes that
                # all recorded evaluations are black-box evaluations where
                # continuations or freeze-thaw was not accounted for during optimization
                if "previous_config_id" in config_result.metadata:
                    previous_config_id = config_result.metadata["previous_config_id"]
                    config_cost -= get_cost(previous_config_id)
            else:
                config_cost = config_result.metadata["time_end"] - global_start

        losses.append(config_result.result["loss"])
        costs.append(config_cost)

    return list(np.minimum.accumulate(losses)), costs, max_cost
