from __future__ import annotations

import os

import numpy as np

import neps


def _get_seed_info(
    path: str | os.PathLike,
    seed: str | int | None,
    key_to_extract: str | None = None,
    consider_continuations: bool = False,
    n_workers: int = 1,
) -> tuple[list[float], list[dict], float]:
    """Reads and processes data per seed."""

    if seed is not None:
        path = os.path.join(path, str(seed), "neps_root_directory")

    data = []
    stats, _ = neps.status(path, print_summary=False)
    for config_id, config_result in stats.items():
        info = dict()
        if "info_dict" in config_result.result:
            info = config_result.result["info_dict"]
        data.append((config_id, config_result.result["loss"], info))

    data.sort()
    data = sorted(data, key=lambda x: len(x[0]))

    max_cost = None if key_to_extract == "cost" else 0.0

    if key_to_extract is not None:
        if n_workers == 1:
            # max_cost only relevant for scaling x-axis when using fidelity on the x-axis
            if consider_continuations:
                # calculates continuation costs for MF algorithms
                # NOTE: assumes that all recorded evaluations are black-box evaluations where
                #   continuations or freeze-thaw was not accounted for during optimization
                data.reverse()
                for idx, (data_id, loss, info) in enumerate(data):
                    # `max_cost` tracks the maximum fidelity used for evaluation
                    max_cost = (
                        max(max_cost, info[key_to_extract])
                        if max_cost is not None
                        else None
                    )
                    for _id, _, _info in data[data.index((data_id, loss, info)) + 1 :]:
                        # if `_` is not found in the string, `split()` returns the original
                        # string and the 0-th element is the string itself, which fits the
                        # config ID format for non-NePS optimizers
                        # MF algos in NePS contain a 2-part ID separated by `_` with the first
                        # element denoting config ID and the second element denoting the rung
                        _subset_idx = 1 if "config" in data_id else 0
                        id_config_id = data_id.split("_")[_subset_idx]
                        _id_config_id = _id.split("_")[_subset_idx]
                        # checking if the base config ID is the same
                        if id_config_id != _id_config_id:
                            continue
                        # subtracting the immediate lower fidelity cost available from the
                        # current higher fidelity --> continuation cost
                        info[key_to_extract] -= _info[key_to_extract]
                        data[idx] = (data_id, loss, info)
                        break
                data.reverse()
            else:
                for idx, (data_id, loss, info) in enumerate(data):
                    # `max_cost` tracks the maximum fidelity used for evaluation
                    max_cost = (
                        max(max_cost, info[key_to_extract])
                        if max_cost is not None
                        else None
                    )
        else:
            global_start = data[0][-1]["start_time"]
            max_cost = None if key_to_extract == "cost" else 0
            for idx, (data_id, loss, info) in enumerate(data):
                info["cost"] = info["end_time"] - global_start
                max_cost = max(max_cost, info["cost"]) if max_cost is not None else None
            if max_cost is not None:
                max_cost += 10.0

    losses, infos = list(zip(*data))[1], list(zip(*data))[2]
    if max_cost is None:
        max_cost = 0.0

    return list(losses), list(infos), max_cost


def process_seed(
    path: str | os.PathLike,
    seed: int | str | None,
    key_to_extract: str | None = None,
    consider_continuations: bool = False,
    n_workers: int = 1,
) -> tuple[list, list[float | None] | None, float]:

    losses, infos, max_cost = _get_seed_info(
        path=path,
        seed=seed,
        consider_continuations=consider_continuations,
        n_workers=n_workers,
    )
    incumbent = list(np.minimum.accumulate(losses))
    if key_to_extract is not None:
        cost = [i[key_to_extract] for i in infos]
    else:
        cost = [1.0] * len(infos)

    return incumbent, cost, max_cost
