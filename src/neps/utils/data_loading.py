from __future__ import annotations

import json
import os
import re
from itertools import chain
from typing import Any

import numpy as np
import yaml

import metahyper.api

from .result_utils import get_loss


def read_tasks_and_dev_stages_from_disk(
    paths: list[str],
) -> dict[int | None, dict[Any, Any]]:
    """
    Reads the given tasks and dev stages from the disk.
    :param paths: List of paths to the previous runs.
    :return: dict[task_id, dict[dev_stage, dict[hyperparameter, value]]]
    """
    task_iters = []
    for path in paths:
        task_iters.append(os.scandir(path))
    task_iter = chain.from_iterable(task_iters)
    results: dict[int | None, dict] = {}
    for task_dir in task_iter:
        task_dir_path = task_dir.path
        if not is_valid_task_path(task_dir_path):
            continue
        task_id = get_id_from_path(task_dir_path)
        results[task_id] = {}
        dev_iter = os.scandir(task_dir_path)
        for dev_dir in dev_iter:
            dev_dir_path = dev_dir.path
            if not is_valid_dev_path(dev_dir_path):
                continue
            dev_id = get_id_from_path(dev_dir_path)
            # TODO: Perhaps use 2nd and 3rd argument as well
            result, _, _ = metahyper.api.read(dev_dir_path)
            results[task_id][dev_id] = result
    return results


def read_user_prior_results_from_disk(path: str):
    """
    Reads the user prior results from the disk.
    :param path: Path to the user prior results.
    :return: dict[hyperparameter, value]
    """
    print("prior_path")
    print(path)
    # check if path is valid directory
    if path is None or not os.path.isdir(path):
        return {}
    prior_iter = os.scandir(path)
    results = {}
    for prior_dir in prior_iter:
        prior_dir_path = prior_dir.path
        # check if the path is a directory
        if not os.path.isdir(prior_dir_path):
            continue
        # get name of the directory
        name = os.path.basename(prior_dir_path)
        results[name], _, _ = metahyper.api.read(prior_dir_path)
    return results


def is_valid_task_path(path: str | None):
    """
    Checks if the given path is a valid task path. It follows the pattern task_00000,
    where 00000 is replaced by the task id.
    :param path: path to check.
    :return: True if the path is valid, False otherwise.
    """
    if path is None:
        return False
    pattern = re.compile(r".*task_\d+")
    return pattern.fullmatch(path) is not None and os.path.isdir(path)


def is_valid_dev_path(path: str | None):
    """
    Checks if the given path is a valid path to development stages. It follows the pattern
    task_00000/dev_00000, where 00000 is replaced by the task and development stage ids.
    :param path: path to check.
    :return: True if the path is valid, False otherwise.
    """
    if path is None:
        return False
    # TODO: Test for \ and | in the path, not only any non-alphanumerical character.
    #  Currently, false positives are possible.
    #  This regex expression does not work: ".*task_\d+[\/\\]dev_\d+"
    pattern = re.compile(r".*task_\d+\Wdev_\d+")
    return pattern.fullmatch(path) is not None and os.path.isdir(path)


def is_valid_seed_path(path: str | None):
    """
    Checks if the given path is a valid path to a seed. It follows the pattern seed_00000,
    where 00000 is replaced by the seed.
    :param path: path to check.
    :return: True if the path is valid, False otherwise.
    """
    if path is None:
        return False
    if not os.path.isdir(path):
        return False
    # check if the directory name starts with "seed"
    if not path.split(os.sep)[-1].startswith("seed"):
        return False
    return True
    # pattern = re.compile(r".*seed=\d+")
    # return pattern.fullmatch(path) is not None and os.path.isdir(path)


def get_id_from_path(path: str | None) -> int | None:
    """
    Extracts the id from the given path. The id is the last part of the path, which is a
    multiple digit number.
    :param path: path to extract the id from.
    :return: id as integer or None if no id could be extracted.
    """
    if path is None:
        return None
    numbers = re.findall(r"\d+", path)
    if len(numbers) == 0:
        return None
    number = int(numbers[-1])
    return number


# TODO: Implement summarize results for nested working directories with multiple experiments
def summarize_results(
    working_dir: str,
    final_task_id: int | None = None,
    final_dev_id: int | None = None,
    sub_dir: str = "",
    write_to_file: bool = True,
):
    """
    Summarizes the results of the given working directory. This includes runs over
    multiple seeds. The results are saved in the working directory.
    :param working_dir: path to the working directory that contains directories for all
                        seeds
    :param final_task_id: id of the tasks whose results should be summarized. If None, all tasks are summarized.
    :param final_dev_id: if of the development stage whose results should be summarized. If None, all development stages are summarized.
    :return: None
    """
    best_losses = []
    seed_iter = os.scandir(working_dir)
    for seed_dir in seed_iter:
        seed_dir_path = seed_dir.path
        if not is_valid_seed_path(seed_dir_path):
            continue
        print("seed: ", seed_dir.path)
        seed_dir_path = os.path.join(seed_dir_path, sub_dir)
        if final_task_id is not None and final_dev_id is not None:
            results = read_tasks_and_dev_stages_from_disk([seed_dir_path])
            #  TOOD: only use IDs if provided
            final_results = results[final_task_id][final_dev_id]
        else:
            final_results, _, _ = metahyper.api.read(seed_dir_path)

        # This part is copied from neps.status()
        best_loss = float("inf")
        # TODO: This is not used, remove it?
        # best_config_id = None
        # best_config = None
        num_error = 0
        for _, evaluation in final_results.items():
            if evaluation.result == "error":
                num_error += 1
            loss = get_loss(evaluation.result, ignore_errors=True)
            if isinstance(loss, float) and loss < best_loss:
                best_loss = get_loss(evaluation.result)
                # best_config = evaluation.config
                # best_config_id = config_id
        best_losses.append(best_loss)
    best_losses_metrics = {}
    print(best_losses)
    best_losses_metrics["best_loss_mean"] = float(np.mean(best_losses))
    best_losses_metrics["best_loss_std"] = float(np.std(best_losses))
    best_losses_metrics["best_loss_std_err"] = float(
        np.std(best_losses) / np.sqrt(np.size(best_losses))
    )
    best_losses_metrics["best_loss_min"] = float(np.min(best_losses))
    best_losses_metrics["best_loss_max"] = float(np.max(best_losses))
    best_losses_metrics["best_loss_max"] = float(np.max(best_losses))
    best_losses_metrics["best_loss_median"] = float(np.median(best_losses))
    best_losses_metrics["best_loss_quantile_25"] = float(np.quantile(best_losses, 0.25))
    best_losses_metrics["best_loss_quantile_75"] = float(np.quantile(best_losses, 0.75))

    print(best_losses_metrics)
    if write_to_file:
        task_id_str = str(final_task_id).zfill(5)
        dev_id_str = str(final_dev_id).zfill(5)
        file_name = working_dir + "/summary_task_" + task_id_str + "_dev_" + dev_id_str
        with open(file_name + ".yaml", "w") as f:
            yaml.dump(best_losses_metrics, f, default_flow_style=False)
        with open(file_name + ".json", "w") as f:
            f.write(json.dumps(best_losses_metrics))
    return best_losses_metrics


def summarize_results_all_tasks_all_devs(
    path, sub_dir="", file_name="summary", user_prior_dir=None
):
    """
    Summarizes the results of all tasks and all development stages. This includes runs over
    multiple seeds. The results are saved in the working directory.
    :return: None
    """
    # go into the first seed directory and read the tasks and dev stages
    seed_iter = os.scandir(path)
    results = None
    for seed_dir in seed_iter:
        seed_dir_path = seed_dir.path
        if not is_valid_seed_path(seed_dir_path):
            continue
        seed_dir_path = os.path.join(seed_dir_path, sub_dir)
        results = read_tasks_and_dev_stages_from_disk([seed_dir_path])
        break
    summary = {}
    # iterate over all tasks and dev stages
    for task_id, task in results.items():
        for dev_id, _ in task.items():
            summary[(task_id, dev_id)] = summarize_results(
                path,
                final_task_id=task_id,
                final_dev_id=dev_id,
                sub_dir=sub_dir,
                write_to_file=False,
            )
    # with open(os.path.join(path, file_name) + ".yaml", "w") as f:
    #     yaml.dump(summary, f, default_flow_style=False)
    summary_user_prior = {}
    if user_prior_dir is not None:
        user_prior_results = read_user_prior_results_from_disk(
            os.path.join(sub_dir, user_prior_dir)
        )
        for prior_name, _ in user_prior_results.items():
            summary_user_prior[prior_name] = summarize_results(
                working_dir=path,
                sub_dir=os.path.join(sub_dir, user_prior_dir, prior_name),
                write_to_file=False,
            )
    with open(os.path.join(path, file_name) + ".jsonl", "w") as f:
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
