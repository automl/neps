from __future__ import annotations

import os

import numpy as np
import yaml  # type: ignore

from ..utils.common import AttrDict

SINGLE_FIDELITY_ALGORITHMS = [
    "random_search",
    "bayesian_optimization",
]


def load_yaml(filename: str) -> AttrDict:
    with open(filename, encoding="UTF-8") as f:
        # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        args = yaml.load(f, Loader=yaml.FullLoader)
    return AttrDict(args)


def _get_info_neps(path: str | os.PathLike, seed: str | int | None):
    if seed is not None:
        path = os.path.join(path, str(seed), "neps_root_directory")

    with open(
        os.path.join(path, "all_losses_and_configs.txt"),
        encoding="UTF-8",
    ) as f:
        data = f.readlines()
    losses = [
        float(entry.strip().split("Loss: ")[1]) for entry in data if "Loss: " in entry
    ]

    config_ids = [
        f"config_{entry.strip().split('Config ID: ')[1]}"
        for entry in data
        if "Config ID: " in entry
    ]
    info = []
    result_path = os.path.join(path, "results")
    for config_id in config_ids:
        result_yaml = load_yaml(os.path.join(result_path, config_id, "result.yaml"))
        if "info_dict" in result_yaml:
            start_time = (
                result_yaml.info_dict["start_time"]
                if "start_time" in result_yaml.info_dict
                else 0.0
            )
            end_time = (
                result_yaml.info_dict["end_time"]
                if "start_time" in result_yaml.info_dict
                else 0.0
            )
            info.append(
                dict(
                    fidelity=result_yaml.info_dict["fidelity"],
                    cost=result_yaml.info_dict["cost"],
                    start_time=start_time,
                    end_time=end_time,
                    config_id=config_id,
                )
            )
        else:
            info.append(dict())

    data = list(zip(config_ids, losses, info))  # type: ignore

    return data


def _get_seed_info(
    path: str | os.PathLike,
    seed: str | int | None,
    key_to_extract: str | None = None,
    algorithm: str = "random_search",
    n_workers: int = 1,
) -> tuple[list[float], list[dict], float]:
    """Reads and processes data per seed.

    An `algorithm` needs to be passed to calculate continuation costs.
    """

    data = _get_info_neps(path, seed)

    max_cost = None if key_to_extract == "cost" else 0.0
    if key_to_extract is not None:
        if n_workers == 1:
            # max_cost only relevant for scaling x-axis when using fidelity on the x-axis
            if algorithm not in SINGLE_FIDELITY_ALGORITHMS:
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

    data = [(d[1], d[2]) for d in data]
    losses, infos = zip(*data)
    if max_cost is None:
        max_cost = 0.0

    return list(losses), list(infos), max_cost


def process_seed(
    path: str | os.PathLike,
    seed: int | str | None,
    algorithm: str,
    key_to_extract: str | None = None,
    n_workers: int = 1,
) -> tuple[list, list[float | None] | None, float]:

    # `algorithm` is passed to calculate continuation costs
    losses, infos, max_cost = _get_seed_info(
        path,
        seed,
        algorithm=algorithm,
        n_workers=n_workers,
    )
    incumbent = list(np.minimum.accumulate(losses))
    if key_to_extract is not None:
        cost = [i[key_to_extract] for i in infos]
    else:
        cost = [1.0 for _ in infos]

    return incumbent, cost, max_cost
