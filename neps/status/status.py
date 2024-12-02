"""Functions to get the status of a run and save the status to CSV files."""

# ruff: noqa: T201
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from neps.state.filebased import FileLocker
from neps.state.neps_state import NePSState
from neps.state.trial import Trial
from neps.utils.types import ConfigID, _ConfigResultForStats

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace


def get_summary_dict(
    root_directory: str | Path,
    *,
    add_details: bool = False,
) -> dict[str, Any]:
    """Create a dict that summarizes a run.

    Args:
        root_directory: The root directory given to neps.run.
        add_details: If true, add detailed dicts for previous_results, pending_configs,
            and pending_configs_free.

    Returns:
        summary_dict: Information summarizing a run
    """
    root_directory = Path(root_directory)

    # NOTE: We don't lock the shared state since we are just reading and don't need to
    # make decisions based on the state
    shared_state = NePSState.create_or_load(root_directory, load_only=True)
    trials = shared_state.lock_and_read_trials()

    evaluated: dict[ConfigID, _ConfigResultForStats] = {}

    for trial in trials.values():
        if trial.report is None:
            continue

        _result_for_stats = _ConfigResultForStats(
            id=trial.id,
            config=trial.config,
            result=trial.report.to_deprecate_result_dict(),
            metadata=asdict(trial.metadata),
        )
        evaluated[trial.id] = _result_for_stats

    in_progress = {
        trial.id: trial.config
        for trial in trials.values()
        if trial.State == Trial.State.EVALUATING
    }
    pending = {
        trial.id: trial.config
        for trial in trials.values()
        if trial.State == Trial.State.PENDING
    }

    summary: dict[str, Any] = {}

    if add_details:
        summary["previous_results"] = evaluated
        summary["pending_configs"] = {**in_progress, **pending}
        summary["pending_configs_free"] = pending

    summary["num_evaluated_configs"] = len(evaluated)
    summary["num_pending_configs"] = len(in_progress) + len(pending)
    summary["num_pending_configs_with_worker"] = len(in_progress)

    summary["best_loss"] = float("inf")
    summary["best_config_id"] = None
    summary["best_config_metadata"] = None
    summary["best_config"] = None
    summary["num_error"] = 0
    for evaluation in evaluated.values():
        if evaluation.result == "error":
            summary["num_error"] += 1
        loss = evaluation.loss
        if isinstance(loss, float) and loss < summary["best_loss"]:
            summary["best_loss"] = loss
            summary["best_config"] = evaluation.config
            summary["best_config_id"] = evaluation.id
            summary["best_config_metadata"] = evaluation.metadata

    return summary


def status(
    root_directory: str | Path,
    *,
    best_losses: bool = False,
    best_configs: bool = False,
    all_configs: bool = False,
    print_summary: bool = True,
) -> tuple[dict[str, _ConfigResultForStats], dict[str, SearchSpace]]:
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
            f"#Pending configs with worker: {summary['num_pending_configs_with_worker']}",
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


def _initiate_summary_csv(root_directory: str | Path) -> tuple[Path, Path, FileLocker]:
    """Initializes a summary CSV and an associated locker for file access control.

    Args:
        root_directory: The root directory where the summary CSV directory,
            containing CSV files and a locker for file access control, will be created.

    Returns:
        A tuple containing the file paths for the configuration data CSV, run data CSV,
        and a locker file.

    The locker is used for file access control to ensure data integrity in a
    multi-threaded or multi-process environment.
    """
    root_directory = Path(root_directory)
    summary_csv_directory = Path(root_directory / "summary_csv")
    summary_csv_directory.mkdir(parents=True, exist_ok=True)

    csv_config_data = summary_csv_directory / "config_data.csv"
    csv_run_data = summary_csv_directory / "run_status.csv"

    csv_locker = FileLocker(summary_csv_directory / ".csv_lock", poll=2, timeout=600)

    return (
        csv_config_data,
        csv_run_data,
        csv_locker,
    )


def _get_dataframes_from_summary(
    root_directory: str | Path,
    *,
    include_metadatas: bool = True,
    include_results: bool = True,
    include_configs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate Pandas DataFrames from summary data retrieved from a run.

    Args:
        root_directory (str | Path): The root directory of the NePS run.
        include_metadatas (bool): Include metadata in the DataFrames (Default: True).
        include_results (bool): Include results in the DataFrames (Default: True).
        include_configs (bool): Include configurations in the DataFrames (Default: True).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    indices_prev = []
    config_data_prev = []
    result_data_prev = []
    metadata_data_prev = []

    indices_pending = []
    config_data_pending = []

    summary = get_summary_dict(root_directory=root_directory, add_details=True)

    for key_prev, config_result_prev in summary["previous_results"].items():
        indices_prev.append(str(key_prev))
        if include_configs:
            config_data_prev.append(config_result_prev.config)
        if include_results:
            result_data_prev.append(config_result_prev.result)
        if include_metadatas:
            metadata_data_prev.append(config_result_prev.metadata)

    for key_pending, config_pending in summary["pending_configs"].items():
        indices_pending.append(str(key_pending))
        if include_configs:
            config_data_pending.append(config_pending)

    # Creating the dataframe for previous config results.
    df_previous = pd.DataFrame({"config_id": indices_prev})
    df_previous["status"] = "complete"
    df_previous = pd.concat(
        [df_previous, pd.json_normalize(config_data_prev).add_prefix("config.")],
        axis=1,
    )
    df_previous = pd.concat(
        [df_previous, pd.json_normalize(metadata_data_prev).add_prefix("metadata.")],
        axis=1,
    )
    df_previous = pd.concat(
        [df_previous, pd.json_normalize(result_data_prev).add_prefix("result.")],
        axis=1,
    )

    # Creating dataframe for pending configs.
    df_pending = pd.DataFrame({"config_id": indices_pending})
    df_pending["status"] = "pending"
    df_pending = pd.concat(
        [df_pending, pd.json_normalize(config_data_pending).add_prefix("config.")],
        axis=1,
    )

    # Concatenate the two DataFrames
    df_config_data = pd.concat([df_previous, df_pending], join="outer", ignore_index=True)

    # Create a dataframe with the specified additional summary data
    additional_data = {
        "num_evaluated_configs": summary["num_evaluated_configs"],
        "num_pending_configs": summary["num_pending_configs"],
        "num_pending_configs_with_worker": summary["num_pending_configs_with_worker"],
        "best_loss": summary["best_loss"],
        "best_config_id": summary["best_config_id"],
        "num_error": summary["num_error"],
    }

    df_run_data = pd.DataFrame.from_dict(
        additional_data,
        orient="index",
        columns=["value"],
    )
    df_run_data.index.name = "description"

    return df_config_data, df_run_data


def _save_data_to_csv(
    config_data_file_path: Path,
    run_data_file_path: Path,
    locker: FileLocker,
    config_data_df: pd.DataFrame,
    run_data_df: pd.DataFrame,
) -> None:
    """Save data as a CSV while acquiring a lock for data integrity.

    This function saves data to CSV files while acquiring a lock to prevent concurrent
    writes. If the lock is acquired, it writes the data to the CSV files and releases the
    lock.

    Args:
        config_data_file_path: The path to the CSV file for configuration data.
        run_data_file_path: The path to the CSV file for additional run data.
        locker: An object for acquiring and releasing a lock to ensure data integrity.
        config_data_df: The DataFrame containing configuration data.
        run_data_df: The DataFrame containing additional run data.
    """
    with locker.lock():
        try:
            pending_configs = run_data_df.loc["num_pending_configs", "value"]
            pending_configs_with_worker = run_data_df.loc[
                "num_pending_configs_with_worker",
                "value",
            ]
            # Represents the last worker
            if int(pending_configs) == 0 and int(pending_configs_with_worker) == 0:
                config_data_df = config_data_df.sort_values(
                    by="result.loss",
                    ascending=True,
                )
                config_data_df.to_csv(config_data_file_path, index=False, mode="w")
                run_data_df.to_csv(run_data_file_path, index=True, mode="w")

            if run_data_file_path.exists():
                prev_run_data_df = pd.read_csv(run_data_file_path)
                prev_run_data_df = prev_run_data_df.set_index("description")

                num_evaluated_configs_csv = prev_run_data_df.loc[
                    "num_evaluated_configs",
                    "value",
                ]
                num_evaluated_configs_run = run_data_df.loc[
                    run_data_df.index == "num_evaluated_configs",
                    "value",
                ]
                # check if the current worker has more evaluated configs than the previous
                if int(num_evaluated_configs_csv) < num_evaluated_configs_run.iloc[0]:
                    config_data_df = config_data_df.sort_values(
                        by="result.loss",
                        ascending=True,
                    )
                    config_data_df.to_csv(config_data_file_path, index=False, mode="w")
                    run_data_df.to_csv(run_data_file_path, index=True, mode="w")
            # Represents the first worker to be evaluated
            else:
                config_data_df = config_data_df.sort_values(
                    by="result.loss",
                    ascending=True,
                )
                config_data_df.to_csv(config_data_file_path, index=False, mode="w")
                run_data_df.to_csv(run_data_file_path, index=True, mode="w")
        except Exception as e:
            raise RuntimeError(f"Error during data saving: {e}") from e


def post_run_csv(root_directory: str | Path) -> tuple[Path, Path]:
    """Create CSV files summarizing the run data.

    Args:
        root_directory: The root directory of the NePS run.

    Returns:
        The paths to the configuration data CSV and the run data CSV.
    """
    csv_config_data, csv_rundata, csv_locker = _initiate_summary_csv(root_directory)
    csv_config_data = Path(csv_config_data).absolute().resolve()
    csv_rundata = Path(csv_rundata).absolute().resolve()

    df_config_data, df_run_data = _get_dataframes_from_summary(
        root_directory,
        include_metadatas=True,
        include_results=True,
        include_configs=True,
    )

    _save_data_to_csv(
        csv_config_data,
        csv_rundata,
        csv_locker,
        df_config_data,
        df_run_data,
    )
    return csv_config_data, csv_rundata
