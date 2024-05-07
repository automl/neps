"""Utility functions for loading data from disk."""

from __future__ import annotations

import json
import os
import re
from itertools import chain
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import yaml

from neps.runtime import ErrorReport, SharedState
from utils.types import ERROR, ResultDict, _ConfigResultForStats


def _get_loss(
    result: ERROR | ResultDict | float,
    loss_value_on_error: float | None = None,
    *,
    ignore_errors: bool = False,
) -> ERROR | float:
    if result == "error":
        if ignore_errors:
            return "error"

        if loss_value_on_error is not None:
            return loss_value_on_error

        raise ValueError(
            "An error happened during the execution of your run_pipeline function."
            " You have three options: 1. If the error is expected and corresponds to"
            " a loss value in your application (e.g., 0% accuracy), you can set"
            " loss_value_on_error to some float. 2. If sometimes your pipeline"
            " crashes randomly, you can set ignore_errors=True. 3. Fix your error."
        )

    if isinstance(result, dict):
        return float(result["loss"])

    assert isinstance(result, float)
    return float(result)


def _get_cost(
    result: str | dict | float,
    cost_value_on_error: float | None = None,
    *,
    ignore_errors: bool = False,
) -> float | Any:
    if result == "error":
        if ignore_errors:
            return "error"

        if cost_value_on_error is None:
            raise ValueError(
                "An error happened during the execution of your run_pipeline function."
                " You have three options: 1. If the error is expected and corresponds to"
                " a cost value in your application, you can set"
                " cost_value_on_error to some float. 2. If sometimes your pipeline"
                " crashes randomly, you can set ignore_errors=True. 3. Fix your error."
            )

        return cost_value_on_error

    if isinstance(result, dict):
        return float(result["cost"])

    return float(result)


def _get_learning_curve(
    result: str | dict | float,
    learning_curve_on_error: list[float] | float | None = None,
    *,
    ignore_errors: bool = False,
) -> list[float] | Any:
    if result == "error":
        if ignore_errors:
            return "error"

        if learning_curve_on_error is None:
            raise ValueError(
                "An error happened during the execution of your run_pipeline function."
                " You have three options: 1. If the error is expected and corresponds to"
                " a learning curve value in your application, you can set"
                " learning_curve_on_error to some float or list of floats."
                " 2. If sometimes your pipeline"
                " crashes randomly, you can set ignore_errors=True. 3. Fix your error."
            )

        if isinstance(learning_curve_on_error, float):
            learning_curve_on_error = [learning_curve_on_error]

        return learning_curve_on_error

    if isinstance(result, dict):
        return result["info_dict"]["learning_curve"]

    return float(result)


def read_tasks_and_dev_stages_from_disk(
    paths: list[str | Path],
) -> dict[int, dict[int, dict[str, _ConfigResultForStats]]]:
    """Reads the given tasks and dev stages from the disk.

    Args:
        paths: List of paths to the previous runs.

    Returns:
        dict[task_id, dict[dev_stage, dict[config_id, ConfigResult]].
    """
    path_iter = chain.from_iterable(Path(path).iterdir() for path in paths)

    results: dict[int, dict[int, dict[str, _ConfigResultForStats]]] = {}

    for task_dir_path in path_iter:
        if not is_valid_task_path(task_dir_path):
            continue

        task_id = get_id_from_path(task_dir_path)
        if task_id is None:
            continue

        results[task_id] = {}

        for dev_dir_path in task_dir_path.iterdir():
            if not is_valid_dev_path(dev_dir_path):
                continue

            dev_id = get_id_from_path(dev_dir_path)
            if dev_id is None:
                continue

            state = SharedState(Path(dev_dir_path))
            state.update_from_disk()
            result = {
                _id: _ConfigResultForStats(
                    _id,
                    report.config,
                    "error" if isinstance(report, ErrorReport) else report.results,
                    report.metadata,
                )
                for _id, report in state.evaluated_trials.items()
            }
            results[task_id][dev_id] = result

    return results


def read_user_prior_results_from_disk(
    path: str | Path,
) -> dict[str, dict[str, _ConfigResultForStats]]:
    """Reads the user prior results from the disk.

    Args:
        path: Path to the user prior results.

    Returns:
        dict[prior_dir_name, dict[config_id, ConfigResult]].
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Path '{path}' is not a directory.")

    results = {}
    for prior_dir in path.iterdir():
        if not prior_dir.is_dir():
            continue

        state = SharedState(prior_dir)
        with state.sync():
            results[prior_dir.name] = {
                _id: _ConfigResultForStats(
                    _id,
                    report.config,
                    "error" if isinstance(report, ErrorReport) else report.results,
                    report.metadata,
                )
                for _id, report in state.evaluated_trials.items()
            }

    return results


_VALID_TASK_PATH_PATTERN = re.compile(r".*task_\d+")


def is_valid_task_path(path: str | Path | None) -> bool:
    """Checks if the given path is a valid task path.

    It follows the pattern task_00000, where 00000 is replaced by the task id.
    """
    if path is None:
        return False

    return (
        _VALID_TASK_PATH_PATTERN.fullmatch(str(path)) is not None and Path(path).is_dir()
    )


def is_valid_dev_path(path: str | Path | None) -> bool:
    """Checks if the given path is a valid path to development stages.

    It follows the pattern task_00000/dev_00000, where 00000 is replaced by the
    task and development stage ids.
    """
    if path is None:
        return False

    # TODO: Test for \ and | in the path, not only any non-alphanumerical character.
    #  Currently, false positives are possible.
    #  This regex expression does not work: ".*task_\d+[\/\\]dev_\d+"
    pattern = re.compile(r".*task_\d+\Wdev_\d+")
    return pattern.fullmatch(str(path)) is not None and Path(path).is_dir()


def is_valid_seed_path(path: str | Path | None) -> bool:
    """Checks if the given path is a valid path to a seed.

    It follows the pattern seed_00000, where 00000 is replaced by the seed.
    """
    if path is None:
        return False
    path = Path(path)

    if not path.is_dir():
        return False

    return path.name.startswith("seed")


def get_id_from_path(path: str | Path | None) -> int | None:
    """Extracts the id from the given path.

    The id is the last part of the path, which is a multiple digit number.

    Note:
        I think this refers to task ids and not config ids!!!
    """
    if path is None:
        return None
    numbers = re.findall(r"\d+", str(path))
    if len(numbers) == 0:
        return None

    return int(numbers[-1])


class BestLossesDict(TypedDict):
    """Summary of the best losses over multiple seeds."""

    best_loss_mean: float
    best_loss_std: float
    best_loss_std_err: float
    best_loss_min: float
    best_loss_max: float
    best_loss_median: float
    best_loss_quantile_25: float
    best_loss_quantile_75: float


# TODO(unknown): Implement summarize results for nested working directories
# with multiple experiments
def summarize_results(
    working_dir: str | Path,
    final_task_id: int | None = None,
    final_dev_id: int | None = None,
    sub_dir: str = "",
    *,
    write_to_file: bool = True,
) -> BestLossesDict:
    """Summarizes the results of the given working directory.

    This includes runs over multiple seeds.
    The results are saved in the working directory.

    Args:
        working_dir: path to the working directory that contains directories for all seeds
        final_task_id: id of the tasks whose results should be summarized.
            If None, all tasks are summarized.
        final_dev_id: if of the development stage whose results should be summarized.
            If None, all development stages are summarized.
        sub_dir: subdirectory to look into for specific seeds.
            * If subdir is provided: `working_dir/something/<subdir>`
            * Otherwise: `working_dir/something`
        write_to_file: if True, the results are written to a file in the working
            directory, using the latest taks and dev stage ids.
            `summary_task_<task_id>_dev_<dev_id>.yaml`
    """
    working_dir = Path(working_dir)

    best_losses = []
    for seed_dir in working_dir.iterdir():
        if not is_valid_seed_path(seed_dir):
            continue

        if sub_dir:
            seed_dir = seed_dir / sub_dir  # noqa: PLW2901

        if final_task_id is not None and final_dev_id is not None:
            results = read_tasks_and_dev_stages_from_disk([seed_dir])

            # TODO(unknown): only use IDs if provided
            final_results = results[final_task_id][final_dev_id]
        else:
            state = SharedState(Path(seed_dir))
            with state.sync():
                final_results = {
                    _id: _ConfigResultForStats(
                        _id,
                        report.config,
                        "error" if isinstance(report, ErrorReport) else report.results,
                        report.metadata,
                    )
                    for _id, report in state.evaluated_trials.items()
                }

        # This part is copied from neps.status()
        best_loss: float = float("inf")
        num_error = 0
        for _, evaluation in final_results.items():
            if evaluation.result == "error":
                num_error += 1
            loss = _get_loss(evaluation.result, ignore_errors=True)
            if isinstance(loss, float) and loss < best_loss:
                best_loss = loss

        best_losses.append(best_loss)

    if len(best_losses) == 0:
        raise ValueError(f"No results found in directort {working_dir}.")

    best_losses_dict = BestLossesDict(
        best_loss_mean=float(np.mean(best_losses)),
        best_loss_std=float(np.std(best_losses)),
        best_loss_std_err=float(np.std(best_losses) / np.sqrt(np.size(best_losses))),
        best_loss_min=float(np.min(best_losses)),
        best_loss_max=float(np.max(best_losses)),
        best_loss_median=float(np.median(best_losses)),
        best_loss_quantile_25=float(np.quantile(best_losses, 0.25)),
        best_loss_quantile_75=float(np.quantile(best_losses, 0.75)),
    )

    if write_to_file:
        task_id_str = str(final_task_id).zfill(5)
        dev_id_str = str(final_dev_id).zfill(5)
        file_path = working_dir / ("summary_task_" + task_id_str + "_dev_" + dev_id_str)

        with file_path.with_suffix(".yaml").open("w") as f:
            yaml.dump(best_losses_dict, f, default_flow_style=False)

        with file_path.with_suffix(".json").open("w") as f:
            json.dump(best_losses_dict, f)

    return best_losses_dict


def summarize_results_all_tasks_all_devs(
    path: str | Path,
    sub_dir: str = "",
    file_name: str = "summary",
    user_prior_dir: str | Path | None = None,
) -> Any:
    """Summarizes the results of all tasks and all development stages.

    This includes runs overrmultiple seeds. The results are saved in
    the working directory.
    """
    # go into the first seed directory and read the tasks and dev stages
    path = Path(path)
    os.scandir(path)

    # TODO(eddiebergman): Please see issue #80
    for seed_dir in path.iterdir():
        if not is_valid_seed_path(seed_dir):
            continue

        seed_dir_path = seed_dir / sub_dir if sub_dir else seed_dir
        results = read_tasks_and_dev_stages_from_disk([seed_dir_path])
        break
    else:
        raise ValueError(f"No results found in directory {path}.")

    summary = {}
    for task_id, task in results.items():
        for dev_id, _ in task.items():
            summary[(task_id, dev_id)] = summarize_results(
                path,
                final_task_id=task_id,
                final_dev_id=dev_id,
                sub_dir=sub_dir,
                write_to_file=False,
            )

    summary_user_prior = {}
    # TODO(eddiebergman): Please see issue #80, figure out what user_prior_dir is
    if user_prior_dir is not None:
        user_prior_dir = Path(user_prior_dir)

        if sub_dir:
            previously_inferred_path = os.path.join(sub_dir, str(user_prior_dir))  # noqa: PTH118
            raise NotImplementedError(
                "Sorry, don't know what should have been done here but we now explicitly"
                "raise instead of silently summarizing what would be a non-existant path"
                f"before. Previously inferred path was: {previously_inferred_path}"
            )

        user_prior_results = read_user_prior_results_from_disk(user_prior_dir)
        for prior_name, _ in user_prior_results.items():
            summary_user_prior[prior_name] = summarize_results(
                working_dir=path,
                sub_dir=str(user_prior_dir / prior_name),
                write_to_file=False,
            )

    with (path / file_name).with_suffix(".jsonl").open("w") as f:
        # write jsonl file with one line per task and dev stage
        for (task_id, dev_id), metrics in summary.items():
            f.write(
                json.dumps(
                    {"IDs": {"task_id": task_id, "dev_id": dev_id}, "metrics": metrics}
                )
            )
            f.write("\n")
        for prior_name, metrics in summary_user_prior.items():
            f.write(json.dumps({"IDs": {"prior_name": prior_name}, "metrics": metrics}))
            f.write("\n")
