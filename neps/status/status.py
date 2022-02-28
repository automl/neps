from __future__ import annotations

import logging
from pathlib import Path

from metahyper import read
from metahyper.api import ConfigResult

from ..search_spaces.search_space import SearchSpace
from ..utils.result_utils import get_loss


def status(
    working_directory: str | Path,
    best_losses: bool = False,
    best_configs: bool = False,
    all_configs: bool = False,
) -> tuple[dict[str, ConfigResult], dict[str, SearchSpace]]:
    """Print status information of a neps run and return results.

    Args:
        working_directory: The working directory given to neps.run.
        best_losses: If true, show the trajectory of the best loss across evaluations
        best_configs: If true, show the trajectory of the best configs and their losses
            across evaluations
        all_configs: If true, show all configs and their losses

    Returns:
        previous_results: Already evaluated configurations and results.
        pending_configs: Configs that have been sampled, but have not finished evaluating
    """
    working_directory = Path(working_directory)

    previous_results, pending_configs, pending_configs_free = read(
        working_directory, logging.getLogger("neps.status")
    )
    print(f"#Evaluated configs: {len(previous_results)}")
    print(f"#Pending configs: {len(pending_configs)}")
    print(
        f"#Pending configs with worker: {len(pending_configs) - len(pending_configs_free)}"
    )

    if len(previous_results) == 0:
        return previous_results, pending_configs

    best_loss = float("inf")
    best_config_id = None
    best_config = None
    num_error = 0
    for config_id, evaluation in previous_results.items():
        if evaluation.result == "error":
            num_error += 1
        if get_loss(evaluation.result) < best_loss:
            best_loss = get_loss(evaluation.result)
            best_config = evaluation.config
            best_config_id = config_id

    print(f"#Crashed configs: {num_error}")

    print()
    print(f"Best loss: {best_loss}")
    print(f"Best config id: {best_config_id}")
    print(f"Best config: {best_config}")

    if best_losses:
        print()
        print("Best loss across evaluations:")
        best_loss_trajectory = working_directory / "best_loss_trajectory.txt"
        print(best_loss_trajectory.read_text(encoding="utf-8"))

    if best_configs:
        print()
        print("Best configs and their losses across evaluations:")
        print(79 * "-")
        best_loss_config = working_directory / "best_loss_with_config_trajectory.txt"
        print(best_loss_config.read_text(encoding="utf-8"))

    if all_configs:
        print()
        print("All evaluated configs and their losses:")
        print(79 * "-")
        all_loss_config = working_directory / "all_losses_and_configs.txt"
        print(all_loss_config.read_text(encoding="utf-8"))

    return previous_results, pending_configs
