from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from metahyper import read
from metahyper.api import ConfigResult

from ..search_spaces.search_space import SearchSpace
from ..utils.result_utils import get_loss


def get_summary_dict(
    root_directory: str | Path, add_details: bool = False
) -> dict[str, Any]:
    """Create dict that summarizes a run.

    Args:
        root_directory: The root directory given to neps.run.
        add_details: If true, add detailed dicts for previous_results, pending_configs,
            and pending_configs_free.

    Returns:
        summary_dict: Information summarizing a run
    """
    root_directory = Path(root_directory)
    previous_results, pending_configs, pending_configs_free = read(
        root_directory, None, logging.getLogger("neps.status")
    )
    summary = dict()

    if add_details:
        summary["previous_results"] = previous_results
        summary["pending_configs"] = pending_configs
        summary["pending_configs_free"] = pending_configs_free

    summary["num_evaluated_configs"] = len(previous_results)
    summary["num_pending_configs"] = len(pending_configs)
    summary["num_pending_configs_with_worker"] = len(pending_configs) - len(
        pending_configs_free
    )

    summary["best_loss"] = float("inf")
    summary["best_config_id"] = None
    summary["best_config_metadata"] = None
    summary["best_config"] = None
    summary["num_error"] = 0
    for config_id, evaluation in previous_results.items():
        if evaluation.result == "error":
            summary["num_error"] += 1
        loss = get_loss(evaluation.result, ignore_errors=True)
        if isinstance(loss, float) and loss < summary["best_loss"]:
            summary["best_loss"] = get_loss(evaluation.result)
            summary["best_config"] = evaluation.config
            summary["best_config_id"] = config_id
            summary["best_config_metadata"] = evaluation.metadata

    return summary


def status(
    root_directory: str | Path,
    best_losses: bool = False,
    best_configs: bool = False,
    all_configs: bool = False,
    print_summary: bool = True,
) -> tuple[dict[str, ConfigResult], dict[str, SearchSpace]]:
    """Print status information of a neps run and return results.

    Args:
        root_directory: The root directory given to neps.run.
        best_losses: If true, show the trajectory of the best loss across evaluations
        best_configs: If true, show the trajectory of the best configs and their losses
            across evaluations
        all_configs: If true, show all configs and their losses
        print_summary: If true, print a summary of the current run state

    Returns:
        previous_results: Already evaluated configurations and results.
        pending_configs: Configs that have been sampled, but have not finished evaluating
    """
    root_directory = Path(root_directory)
    summary = get_summary_dict(root_directory, add_details=True)

    if print_summary:
        print(f"#Evaluated configs: {summary['num_evaluated_configs']}")
        print(f"#Pending configs: {summary['num_pending_configs']}")
        print(
            f"#Pending configs with worker: {summary['num_pending_configs_with_worker']}"
        )

        print(f"#Crashed configs: {summary['num_error']}")

        if len(summary["previous_results"]) == 0:
            return summary["previous_results"], summary["pending_configs"]

        print()
        print(f"Best loss: {summary['best_loss']}")
        print(f"Best config id: {summary['best_config_id']}")
        print(f"Best config: {summary['best_config']}")

        if best_losses:
            print()
            print("Best loss across evaluations:")
            best_loss_trajectory = root_directory / "best_loss_trajectory.txt"
            print(best_loss_trajectory.read_text(encoding="utf-8"))

        if best_configs:
            print()
            print("Best configs and their losses across evaluations:")
            print(79 * "-")
            best_loss_config = root_directory / "best_loss_with_config_trajectory.txt"
            print(best_loss_config.read_text(encoding="utf-8"))

        if all_configs:
            print()
            print("All evaluated configs and their losses:")
            print(79 * "-")
            all_loss_config = root_directory / "all_losses_and_configs.txt"
            print(all_loss_config.read_text(encoding="utf-8"))

    return summary["previous_results"], summary["pending_configs"]
