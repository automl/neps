"""Utility functions for reading and processing results."""

from __future__ import annotations

from pathlib import Path

import neps


def process_seed(
    *,
    path: str | Path,
    seed: str | int | None,
    key_to_extract: str | None = None,  # noqa: ARG001
    consider_continuations: bool = False,  # noqa: ARG001
    n_workers: int = 1,  # noqa: ARG001
) -> tuple[list[float], list[float], float]:
    """Reads and processes data per seed."""
    path = Path(path)
    if seed is not None:
        path = path / str(seed) / "neps_root_directory"

    _fulldf, _summary = neps.status(path, print_summary=False)
    raise NotImplementedError(
        "I'm sorry, I broke this. We now dump all the information neps has available"
        " into the above dataframe `fulldf`."
    )
    # > sorted_stats = sorted(sorted(stats.items()), key=lambda x: len(x[0]))
    # > stats = OrderedDict(sorted_stats)

    # > # max_cost only relevant for scaling x-axis when using fidelity on the x-axis
    # > max_cost: float = -1.0
    # > if key_to_extract == "fidelity":
    # >     # TODO(eddiebergman): This can crash for a number of reasons, namely if the
    # >     # config crased and it's result is an error, or if the `"info_dict"` and/or
    # >     # `key_to_extract` doesn't exist
    # >     max_cost = max(s.result["info_dict"][key_to_extract] for s in stats.values())

    # > global_start = stats[min(stats.keys())].metadata["time_sampled"]

    # > def get_cost(idx: str) -> float:
    # >     if key_to_extract is not None:
    # >         # TODO(eddiebergman): This can crash for a number of reasons, namely if
    # >         # the config crased and it's result is an error, or if the `"info_dict"`
    # >         # and/or `key_to_extract` doesn't exist
    # >         return float(stats[idx].result["info_dict"][key_to_extract])

    # >     return 1.0

    # > losses = []
    # > costs = []

    # > for config_id, config_result in stats.items():
    # >     config_cost = get_cost(config_id)
    # >     if consider_continuations:
    # >         if n_workers == 1:
    # >             # calculates continuation costs for MF algorithms NOTE: assumes that
    # >             # all recorded evaluations are black-box evaluations where
    # >             # continuations or freeze-thaw was not accounted for during opt
    # >             if "previous_config_id" in config_result.metadata:
    # >                 previous_config_id = config_result.metadata["previous_config_id"]
    # >                 config_cost -= get_cost(previous_config_id)
    # >         else:
    # >             config_cost = config_result.metadata["time_end"] - global_start

    # >     # TODO(eddiebergman): Assumes it never crashed and there's a
    # >     # objective_to_minimize available,not fixing now but it should be addressed
    # >     losses.append(config_result.result["objective_to_minimize"])  # type: ignore
    # >     costs.append(config_cost)

    # > return list(np.minimum.accumulate(losses)), costs, max_cost
