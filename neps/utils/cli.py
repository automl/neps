# flake8: noqa
"""This module provides a command-line interface (CLI) for NePS."""

from __future__ import annotations

import warnings
from datetime import timedelta, datetime
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import logging
import yaml
from pathlib import Path
from typing import Optional, List
import neps
from neps.state.seed_snapshot import SeedSnapshot
from neps.status.status import post_run_csv
import pandas as pd
from neps.utils.common import dynamic_load_object
from neps.state.neps_state import NePSState
from neps.state.trial import Trial
from neps.exceptions import TrialNotFoundError
from neps.optimizers import load_optimizer
from neps.state.optimizer import BudgetInfo, OptimizationState


def validate_directory(path: Path) -> bool:
    """
    Validates whether the given path exists and is a directory.

    Args:
        path (Path): The path to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not path.exists():
        print(f"Error: The directory '{path}' does not exist.")
        return False
    if not path.is_dir():
        print(f"Error: The path '{path}' exists but is not a directory.")
        return False
    return True


def get_root_directory(args: argparse.Namespace) -> Optional[Path]:
    # Command-line argument handling
    if args.root_directory:
        root_dir = Path(args.root_directory)
        if validate_directory(root_dir):
            return root_dir
        else:
            return None

    # Configuration file handling
    config_path = Path("run_config.yaml").resolve()
    if config_path.exists():
        try:
            with config_path.open("r") as file:
                config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error parsing '{config_path}': {e}")
            return None

        root_directory = config.get("root_directory")
        if root_directory:
            root_directory_path = Path(root_directory)
            if validate_directory(root_directory_path):
                return root_directory_path
            else:
                return None
        else:
            print(
                "Error: The 'run_config.yaml' file exists but does not contain the "
                "'root_directory' key."
            )
            return None
    else:
        print(
            "Error: 'root_directory' must be provided as a command-line argument "
            "or defined in 'run_config.yaml'."
        )
        return None


def init_config(args: argparse.Namespace) -> None:
    """Creates a 'run_args' configuration YAML file template if it does not already
    exist.
    """
    config_path = (
        Path(args.config_path).resolve()
        if args.config_path
        else Path("run_config.yaml").resolve()
    )

    if args.database:
        if config_path.exists():
            with config_path.open("r") as file:
                run_args = yaml.safe_load(file)

            max_cost_total = run_args.get("max_cost_total")
            # Create the optimizer
            _, optimizer_info = load_optimizer(
                optimizer=run_args.get("optimizer"),  # type: ignore
                space=run_args.get("pipeline_space"),  # type: ignore
            )
            try:
                directory = run_args.get("root_directory")
                if directory is None:
                    return
                else:
                    directory = Path(directory)
                is_new = not directory.exists()
                _ = NePSState.create_or_load(
                    path=directory,
                    optimizer_info=optimizer_info,
                    optimizer_state=OptimizationState(
                        seed_snapshot=SeedSnapshot.new_capture(),
                        budget=(
                            BudgetInfo(max_cost_total=max_cost_total, used_cost_budget=0)
                            if max_cost_total is not None
                            else None
                        ),
                        shared_state=None,  # TODO: Unused for the time being...
                    ),
                )
                if is_new:
                    print("NePS state was successfully created.")
                else:
                    print("NePS state was already created.")
            except Exception as e:
                print(f"Error creating neps state: {e}")
        else:
            print(
                f"{config_path} does not exist. Make sure that your configuration "
                f"file already exists if you don't have specified your own path. "
                f"Run 'neps init' to create run_config.yaml"
            )

    elif not config_path.exists():
        with config_path.open("w") as file:
            template = args.template if args.template else "basic"
            if template == "basic":
                file.write(
                    """# Add your NEPS configuration settings here

evaluate_pipeline:
  path: "path/to/your/evaluate_pipeline.py"
  name: name_of_your_pipeline_function

pipeline_space:
  float_parameter_name:
    type: "float"
    lower:
    upper:
    log: false
  int_parameter_name:
    type: "int"
    lower:
    upper:
  categorical_parameter_name:
    choices: ["choice1", "choice2", "choice3"]
  constant_parameter_name: 17

root_directory: "set/path/for/root_directory"
max_evaluations_total:
overwrite_working_directory:
"""
                )
            elif template == "complete":
                file.write(
                    """# Full Configuration Template for NePS

evaluate_pipeline:
  path: path/to/your/evaluate_pipeline.py  # Path to the function file
  name: example_pipeline              # Function name within the file

pipeline_space:
  learning_rate:
    lower: 1e-5
    upper: 1e-1
    log: true
  epochs:
    lower: 5
    upper: 20
    is_fidelity: true
  optimizer:
    choices: [adam, sgd, adamw]
  batch_size: 64

root_directory: path/to/results       # Directory for result storage
max_evaluations_total: 20             # Budget
max_cost_total:

# Debug and Monitoring
overwrite_working_directory: false
post_run_summary: true

# Parallelization Setup
max_evaluations_per_run:
continue_until_max_evaluation_completed: true

# Error Handling
objective_value_on_error:
cost_value_on_error:
ignore_errors:

# Customization Options
optimizer: hyperband       # Internal key to select a NePS optimizer.
"""
                )
    else:
        print(f"Path {config_path} does already exist.")


def parse_kv_pairs(kv_list: list[str]) -> dict:
    """Parse a list of key=value strings into a dictionary with appropriate types."""

    def convert_value(value: str) -> int | float | str:
        """Convert the value to the appropriate type."""
        # Check for boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Check for float if value contains '.' or 'e'
        if "." in value or "e" in value.lower():
            try:
                return float(value)
            except ValueError:
                return value  # Return as string if conversion fails

        # Check for integer
        try:
            return int(value)
        except ValueError:
            return value  # Return as string if conversion fails

    result = {}
    for item in kv_list:
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = convert_value(value)
        else:
            raise ValueError("Each kwarg must be in key=value format.")
    return result


def run_optimization(args: argparse.Namespace) -> None:
    """Collects arguments from the parser and runs the NePS optimization.
    Args: args (argparse.Namespace): Parsed command-line arguments.
    """
    if args.run_args is None:
        run_args = Path("run_config.yaml")
    else:
        run_args = args.run_args

    if isinstance(args.evaluate_pipeline, str):
        module_path, function_name = args.evaluate_pipeline.split(":")
        evaluate_pipeline = dynamic_load_object(module_path, function_name)
    else:
        evaluate_pipeline = args.evaluate_pipeline

    logging.basicConfig(level=logging.INFO)
    with open(run_args, "r") as file:
        run_config = yaml.safe_load(file)

    neps.run(**run_config)


def info_config(args: argparse.Namespace) -> None:
    """Handles the info-config command by providing information based on directory
    and id."""
    directory_path = get_root_directory(args)
    if directory_path is None:
        return
    config_id = args.id

    neps_state = load_neps_state(directory_path)
    if neps_state is None:
        return
    try:
        trial = neps_state.unsafe_retry_get_trial_by_id(config_id)
    except TrialNotFoundError:
        print(f"No trial found with ID {config_id}.")
        return

    print("Trial Information:")
    print(f"  Trial ID: {trial.metadata.id}")
    print(f"  State: {trial.metadata.state}")
    print(f"  Configurations:")
    for key, value in trial.config.items():
        print(f"    {key}: {value}")

    print("\nMetadata:")
    print(f"  Location: {trial.metadata.location}")
    print(f"  Previous Trial ID: {trial.metadata.previous_trial_id}")
    print(f"  Sampling Worker ID: {trial.metadata.sampling_worker_id}")
    print(f"  Time Sampled: {convert_timestamp(trial.metadata.time_sampled)}")
    print(f"  Evaluating Worker ID: {trial.metadata.evaluating_worker_id}")
    print(f"  Evaluation Duration: {format_duration(trial.metadata.evaluation_duration)}")
    print(f"  Time Started: {convert_timestamp(trial.metadata.time_started)}")
    print(f"  Time End: {convert_timestamp(trial.metadata.time_end)}")

    if trial.report is not None:
        print("\nReport:")
        print(f"  Objective_to_minimize: {trial.report.objective_to_minimize}")
        print(f"  Cost: {trial.report.cost}")
        print(f"  Reported As: {trial.report.reported_as}")
        error = trial.report.err
        if error is not None:
            print(f"  Error Type: {type(error).__name__}")
            print(f"  Error Message: {str(error)}")
            print(f"  Traceback:")
            print(f"    {trial.report.tb}")
    else:
        print("No report available.")


def load_neps_errors(args: argparse.Namespace) -> None:
    """Handles the 'errors' command by loading errors from the neps_state."""
    directory_path = get_root_directory(args)
    if directory_path is None:
        return

    neps_state = load_neps_state(directory_path)
    if neps_state is None:
        return
    errors = neps_state.lock_and_get_errors()

    if not errors.errs:
        print("No errors found.")
        return

    # Print out the errors in a human-readable format
    print(f"Loaded Errors from directory: {directory_path}\n")

    for error in errors.errs:
        print(f"Error in Trial ID: {error.trial_id}")
        print(f"  Worker ID: {error.worker_id}")
        print(f"  Error Type: {error.err_type}")
        print(f"  Error Message: {error.err}")
        print(f"  Traceback:")
        print(f"{error.tb}")
        print("\n" + "-" * 50 + "\n")


def sample_config(args: argparse.Namespace) -> None:
    """Handles the sample-config command which samples configurations from the NePS
    state."""
    # Load run_args from the provided path or default to run_config.yaml
    if args.run_args:
        run_args_path = Path(args.run_args)
    else:
        run_args_path = Path("run_config.yaml")

    if not run_args_path.exists():
        print(f"Error: run_args file {run_args_path} does not exist.")
        return

    with run_args_path.open("r") as file:
        run_args = yaml.safe_load(file)

    # Get root_directory from the run_args
    root_directory = run_args.get("root_directory")
    if not root_directory:
        print("Error: 'root_directory' is not specified in the run_args file.")
        return

    root_directory = Path(root_directory)
    if not root_directory.exists():
        print(f"Error: The directory {root_directory} does not exist.")
        return

    neps_state = load_neps_state(root_directory)
    if neps_state is None:
        return

    # Get the worker_id and number_of_configs from arguments
    worker_id = args.worker_id
    num_configs = args.number_of_configs if args.number_of_configs else 1

    optimizer, _ = load_optimizer(
        optimizer=run_args.get("optimizer"),  # type: ignore
        space=run_args.get("pipeline_space"),  # type: ignore
    )

    # Sample trials
    for _ in range(num_configs):
        try:
            trial = neps_state.lock_and_sample_trial(optimizer, worker_id=worker_id)
        except Exception as e:
            print(f"Error during configuration sampling: {e}")
            continue  # Skip to the next iteration

        print(f"Sampled configuration with Trial ID: {trial.id}")
        print(f"Location: {trial.metadata.location}")
        print("Configuration:")
        for key, value in trial.config.items():
            print(f"  {key}: {value}")
        print("\n")


def convert_timestamp(timestamp: float | None) -> str:
    """Convert a UNIX timestamp to a human-readable datetime string."""
    if timestamp is None:
        return "None"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float | None) -> str:
    """Convert duration in seconds to a h:min:sec format."""
    if seconds is None:
        return "None"
    duration = str(timedelta(seconds=seconds))
    # Remove milliseconds for alignment
    if "." in duration:
        duration = duration.split(".")[0]
    return duration


def compute_duration(start_time: float) -> str:
    """Compute duration from start_time to current time."""
    return format_duration(datetime.now().timestamp() - start_time)


def status(args: argparse.Namespace) -> None:
    """Handles the status command, providing a summary of the NEPS run."""
    # Get the root_directory from args or load it from run_config.yaml
    raise NotImplementedError("Sorry, broken for the moment")
    # > directory_path = get_root_directory(args)
    # > if directory_path is None:
    # >     return

    # > neps_state = load_neps_state(directory_path)
    # > if neps_state is None:
    # >     return

    # > summary = get_summary_dict(directory_path, add_details=True)

    # > # Calculate the number of trials in different states
    # > trials = neps_state.lock_and_read_trials()
    # > evaluating_trials_count = sum(
    # >     1 for trial in trials.values() if trial.metadata.state == Trial.State.EVALUATING
    # > )
    # > pending_trials_count = summary["num_pending_configs"]
    # > succeeded_trials_count = summary["num_evaluated_configs"] - summary["num_error"]
    # > failed_trials_count = summary["num_error"]

    # > # Print summary
    # > print("NePS Status:")
    # > print("-----------------------------")
    # > print(f"Optimizer: {neps_state.lock_and_get_optimizer_info().info['optimizer_alg']}")
    # > print(f"Succeeded Trials: {succeeded_trials_count}")
    # > print(f"Failed Trials (Errors): {failed_trials_count}")
    # > print(f"Active Trials: {evaluating_trials_count}")
    # > print(f"Pending Trials: {pending_trials_count}")
    # > print(f"Best Objective_to_minimize Achieved: {summary['best_objective_to_minimize']}")

    # > print("\nLatest Trials:")
    # > print("-----------------------------")

    # > # Retrieve and sort the trials by time_sampled
    # > sorted_trials = sorted(
    # >     trials.values(), key=lambda t: t.metadata.time_sampled, reverse=True
    # > )

    # > # Filter trials based on state
    # > if args.pending:
    # >     filtered_trials = [
    # >         trial for trial in sorted_trials if trial.metadata.state.name == "PENDING"
    # >     ]
    # > elif args.evaluating:
    # >     filtered_trials = [
    # >         trial for trial in sorted_trials if trial.metadata.state.name == "EVALUATING"
    # >     ]
    # > elif args.succeeded:
    # >     filtered_trials = [
    # >         trial for trial in sorted_trials if trial.metadata.state.name == "SUCCESS"
    # >     ]
    # > else:
    # >     filtered_trials = sorted_trials[:7]

    # > # Define column headers with fixed width
    # > header_format = "{:<20} {:<10} {:<10} {:<40} {:<12} {:<10}"
    # > row_format = "{:<20} {:<10} {:<10} {:<40} {:<12} {:<10}"

    # > # Print the header
    # > print(
    # >     header_format.format(
    # >         "Sampled Time",
    # >         "Duration",
    # >         "Trial ID",
    # >         "Worker ID",
    # >         "State",
    # >         "Objective_to_minimize",
    # >     )
    # > )

    # > # Print the details of the filtered trials
    # > for trial in filtered_trials:
    # >     time_sampled = convert_timestamp(trial.metadata.time_sampled)
    # >     if trial.metadata.state.name in ["PENDING", "EVALUATING"]:
    # >         duration = compute_duration(trial.metadata.time_sampled)
    # >     else:
    # >         duration = (
    # >             format_duration(trial.metadata.evaluation_duration)
    # >             if trial.metadata.evaluation_duration
    # >             else "N/A"
    # >         )
    # >     trial_id = trial.id
    # >     worker_id = trial.metadata.sampling_worker_id
    # >     state = trial.metadata.state.name
    # >     objective_to_minimize = (
    # >         f"{trial.report.objective_to_minimize:.6f}"
    # >         if (trial.report and trial.report.objective_to_minimize is not None)
    # >         else "N/A"
    # >     )

    # >     print(
    # >         row_format.format(
    # >             time_sampled, duration, trial_id, worker_id, state, objective_to_minimize
    # >         )
    # >     )

    # > # If no specific filter is applied, print the best trial and optimizer info
    # > if not args.pending and not args.evaluating and not args.succeeded:
    # >     if summary["best_config_id"] is not None:
    # >         print("\nBest Trial:")
    # >         print("-----------------------------")
    # >         print(f"ID: {summary['best_config_id']}")
    # >         print(f"Loss: {summary['best_objective_to_minimize']}")
    # >         print("Config:")
    # >         for key, value in summary["best_config"].items():
    # >             print(f"  {key}: {value}")

    # >         print(
    # >             f"\nMore details: neps info-config {summary['best_config_id']} "
    # >             f"(use --root-directory if not using run_config.yaml)"
    # >         )
    # >     else:
    # >         print("\nBest Trial:")
    # >         print("-----------------------------")
    # >         print("\nNo successful trial found.")

    # >     # Display optimizer information
    # >     optimizer_info = neps_state.lock_and_get_optimizer_info().info
    # >     optimizer_name = optimizer_info.get("optimizer_name", "N/A")
    # >     optimizer_alg = optimizer_info.get("optimizer_alg", "N/A")
    # >     optimizer_args = optimizer_info.get("optimizer_args", {})

    # >     print("\nOptimizer Information:")
    # >     print("-----------------------------")
    # >     print(f"Name: {optimizer_name}")
    # >     print(f"Algorithm: {optimizer_alg}")
    # >     print("Parameter:")
    # >     for arg, value in optimizer_args.items():
    # >         print(f"  {arg}: {value}")

    # >     print("-----------------------------")


def results(args: argparse.Namespace) -> None:
    """Handles the 'results' command by displaying incumbents, optionally plotting,
    and dumping results to files based on the specified options."""
    directory_path = get_root_directory(args)
    if directory_path is None:
        return

    # Attempt to generate the summary CSV
    try:
        csv_config_data_path, _ = post_run_csv(directory_path)
    except Exception as e:
        print(f"Error generating summary CSV: {e}")
        return

    summary_csv_dir = csv_config_data_path.parent  # 'summary_csv' directory

    # Load NePS state
    neps_state = load_neps_state(directory_path)
    if neps_state is None:
        return

    def sort_trial_id(trial_id: str) -> List[int]:
        parts = trial_id.split("_")  # Split the ID by '_'
        # Convert each part to an integer for proper numeric sorting
        return [int(part) for part in parts]

    trials = neps_state.lock_and_read_trials()
    sorted_trials = sorted(trials.values(), key=lambda x: sort_trial_id(x.id))

    # Compute incumbents
    incumbents = compute_incumbents(sorted_trials)
    incumbents_ids = [trial.id for trial in incumbents]

    # Handle Dump Options
    if args.dump_all_configs or args.dump_incumbents:
        if args.dump_all_configs:
            dump_all_configs(csv_config_data_path, summary_csv_dir, args.dump_all_configs)
            return

        if args.dump_incumbents:
            dump_incumbents(
                csv_config_data_path,
                summary_csv_dir,
                args.dump_incumbents,
                incumbents_ids,
            )
            return

    # Display Results
    display_results(directory_path, incumbents)

    # Handle Plotting
    if args.plot:
        plot_path = plot_incumbents(sorted_trials, incumbents, summary_csv_dir)
        print(f"Plot saved to '{plot_path}'.")


def load_neps_state(directory_path: Path) -> Optional[NePSState]:
    """Load the NePS state with error handling."""
    try:
        return NePSState.create_or_load(directory_path, load_only=True)
    except Exception as e:
        print(f"Unexpected error loading NePS state: {e}")
    return None


def compute_incumbents(sorted_trials: List[Trial]) -> List[Trial]:
    """Compute the list of incumbent trials based on the best objective_to_minimize."""
    best_objective_to_minimize = float("inf")
    incumbents = []
    for trial in sorted_trials:
        if (
            trial.report is not None
            and trial.report.objective_to_minimize is not None
            and trial.report.objective_to_minimize < best_objective_to_minimize
        ):
            best_objective_to_minimize = trial.report.objective_to_minimize
            incumbents.append(trial)
    return incumbents[::-1]  # Reverse for most recent first


def dump_all_configs(
    csv_config_data_path: Path, summary_csv_dir: Path, dump_format: str
) -> None:
    """Dump all configurations to the specified format."""
    dump_format = dump_format.lower()
    supported_formats = ["csv", "json", "parquet"]
    if dump_format not in supported_formats:
        print(
            f"Unsupported dump format: '{dump_format}'. "
            f"Supported formats are: {supported_formats}."
        )
        return

    base_name = csv_config_data_path.stem  # 'config_data'

    if dump_format == "csv":
        # CSV is already available
        print(
            f"All trials successfully dumped to '{summary_csv_dir}/{base_name}.{dump_format}'."
        )
    else:
        # Define output file path with desired extension
        output_file_name = f"{base_name}.{dump_format}"
        output_file_path = summary_csv_dir / output_file_name

        try:
            # Read the existing CSV into DataFrame
            df = pd.read_csv(csv_config_data_path)

            # Save to the desired format
            if dump_format == "json":
                df.to_json(output_file_path, orient="records", indent=4)
            elif dump_format == "parquet":
                df.to_parquet(output_file_path, index=False)

            print(f"All trials successfully dumped to '{output_file_path}'.")
        except Exception as e:
            print(f"Error dumping all trials to '{dump_format}': {e}")


def dump_incumbents(
    csv_config_data_path: Path,
    summary_csv_dir: Path,
    dump_format: str,
    incumbents_ids: List[str],
) -> None:
    """Dump incumbent trials to the specified format."""
    dump_format = dump_format.lower()
    supported_formats = ["csv", "json", "parquet"]
    if dump_format not in supported_formats:
        print(
            f"Unsupported dump format: '{dump_format}'. Supported formats are: {supported_formats}."
        )
        return

    base_name = "incumbents"  # Name for incumbents file

    if not incumbents_ids:
        print("No incumbent trials found to dump.")
        return

    try:
        # Read the existing CSV into DataFrame
        df = pd.read_csv(csv_config_data_path)

        # Filter DataFrame for incumbent IDs
        df_incumbents = df[df["config_id"].isin(incumbents_ids)]

        if df_incumbents.empty:
            print("No incumbent trials found in the summary CSV.")
            return

        # Define output file path with desired extension
        output_file_name = f"{base_name}.{dump_format}"
        output_file_path = summary_csv_dir / output_file_name

        # Save to the desired format
        if dump_format == "csv":
            df_incumbents.to_csv(output_file_path, index=False)
        elif dump_format == "json":
            df_incumbents.to_json(output_file_path, orient="records", indent=4)
        elif dump_format == "parquet":
            df_incumbents.to_parquet(output_file_path, index=False)

        print(f"Incumbent trials successfully dumped to '{output_file_path}'.")
    except Exception as e:
        print(f"Error dumping incumbents to '{dump_format}': {e}")


def display_results(directory_path: Path, incumbents: List[Trial]) -> None:
    """Display the results of the NePS run."""
    print(f"Results for NePS run: {directory_path}")
    print("--------------------")
    print("All Incumbent Trials:")
    header = f"{'ID':<6} {'Loss':<12} {'Config':<60}"
    print(header)
    print("-" * len(header))
    if incumbents:
        for trial in incumbents:
            if (
                trial.report is not None
                and trial.report.objective_to_minimize is not None
            ):
                config = ", ".join(f"{k}: {v}" for k, v in trial.config.items())
                print(
                    f"{trial.id:<6} {trial.report.objective_to_minimize:<12.6f} {config:<60}"
                )
            else:
                print(f"Trial {trial.id} has no valid objective_to_minimize.")
    else:
        print("No Incumbent Trials found.")


def plot_incumbents(
    all_trials: List[Trial], incumbents: List[Trial], directory_path: Path
) -> str:
    """Plot the evolution of incumbent trials over the total number of trials."""
    id_to_index = {trial.id: idx + 1 for idx, trial in enumerate(all_trials)}

    # Collect data for plotting
    x_values = [id_to_index[incumbent.id] for incumbent in incumbents]
    y_values = [
        incumbent.report.objective_to_minimize
        for incumbent in incumbents
        if incumbent.report is not None
        and incumbent.report.objective_to_minimize is not None
    ]

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x=x_values,
        y=y_values,
        marker="o",
        linestyle="-",
        markersize=8,
        color="dodgerblue",
    )
    plt.xlabel("Number of Trials")
    plt.ylabel("Loss")
    plt.title("Evolution of Incumbents Over Trials")

    # Dynamically set x-axis ticks based on the number of trials
    num_trials = len(all_trials)
    if num_trials < 20:
        tick_spacing = 1  # Every trial is labeled if fewer than 20 trials
    else:
        tick_spacing = max(
            5, round(num_trials / 10 / 5) * 5
        )  # Round to nearest multiple of 5

    ticks = np.arange(0, num_trials + 1, tick_spacing)
    ticks[0] = 1
    plt.xticks(ticks)

    sns.set_style("whitegrid")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Save the figure
    plot_file_name = "incumbents_evolution.png"
    plot_path = os.path.join(directory_path, plot_file_name)
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def print_help(args: Optional[argparse.Namespace] = None) -> None:
    """Prints help information for the NEPS CLI."""
    help_text = """
Usage: neps [COMMAND] [OPTIONS]

Available Commands:
-------------------

neps init [OPTIONS]
    Generates a 'run_args' YAML template file.
    Options:
    --config-path <path/to/config.yaml> (Optional: Specify the path for the config
    file. Default is run_config.yaml)
    --template [basic|complete] (Optional: Choose between a basic or complete template.)
    --database (Optional: Creates a NEPS state. Requires an existing config.yaml.)

neps run [OPTIONS]
    Runs a neural pipeline search.
    Options:
    --run-args <path_to_run_args> (Path to the YAML configuration file.)
    --run-pipeline <path_to_module:function_name> (Path and function for the pipeline.)
    --pipeline-space <path_to_yaml> (Path to the YAML defining the search space.)
    --root-directory <path> (Optional: Directory for saving progress and
    synchronization. Default is 'root_directory' from run_config.yaml if not provided.)
    --overwrite-working-directory (Deletes the working directory at the start of the run.)
    --post-run-summary/--no-post-run-summary (Toggle summary after running.)
    --max-evaluations-total <int> (Total number of evaluations to run.)
    --max-evaluations-per-run <int> (Max evaluations per run call.)
    --continue-until-max-evaluation-completed (Continue until max evaluations are completed.)
    --max-cost-total <float> (Max cost before halting new evaluations.)
    --ignore-errors (Ignore errors during optimization.)
    --objective_to_minimize-value-on-error <float> (Assumed objective_to_minimize value on error.)
    --cost-value-on-error <float> (Assumed cost value on error.)
    --optimizer <key> (optimizer algorithm key for optimization.)
    --optimizer-kwargs <key=value>... (Additional kwargs for the optimizer.)

neps info-config <id> [OPTIONS]
    Provides detailed information about a specific configuration by its ID.
    Options:
    --root-directory <path> (Optional: Path to your root_directory. Default is
    'root_directory' from run_config.yaml if not provided.)

neps errors [OPTIONS]
    Lists all errors from the specified NePS run.
    Options:
    --root-directory <path> (Optional: Path to your root_directory. Default is
    'root_directory' from run_config.yaml if not provided.)

neps sample-config [OPTIONS]
    Sample a configuration from the existing NePS state.
    Options:
    --root-directory <path> (Optional: Path to your root_directory. Default is
    'root_directory' from run_config.yaml if not provided.)

neps status [OPTIONS]
    Check the status of the NePS run.
    Options:
    --root-directory <path> (Optional: Path to your root_directory. Default is
    'root_directory' from run_config.yaml if not provided.)
    --pending (Show only pending trials.)
    --evaluating (Show only evaluating trials.)
    --succeeded (Show only succeeded trials.)

neps results [OPTIONS]
    Display results of the NePS run.
    Options:
    --root-directory <path> (Optional: Path to your root_directory. Defaults is
    'root_directory' from run_config.yaml if not provided.)
    --plot (Plot the results if set.)

neps help
    Displays this help message.
    """
    print(help_text)


def generate_markdown_from_parser(parser: argparse.ArgumentParser, filename: str) -> None:
    lines = []

    # Add the general parser description
    if parser.description:
        lines.append(f"# {parser.description}")
        lines.append("\n")

    # Extract subparsers (if they exist)
    subcommands = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for cmd, subparser in action.choices.items():
                subcommands[cmd] = subparser

    # Order subcommands: init, run, status, help (always last), followed by any others
    order = ["init", "run", "status"]
    sorted_subcommands = [cmd for cmd in order if cmd in subcommands]
    sorted_subcommands += [
        cmd for cmd in subcommands if cmd not in order and cmd != "help"
    ]
    if "help" in subcommands:
        sorted_subcommands.append("help")

    # Iterate through sorted subcommands and generate the documentation
    for cmd in sorted_subcommands:
        subparser = subcommands[cmd]

        # Command header
        lines.append(f"## **`{cmd}` Command**")
        lines.append("\n")

        # Command description
        if subparser.description:
            lines.append(f"{subparser.description}")
            lines.append("\n")

        # Extract and list options (Required and Optional)
        lines.append("**Arguments:**")
        lines.append("\n")

        required_args = []
        optional_args = []

        for sub_action in subparser._actions:
            option_strings = ", ".join(sub_action.option_strings)
            option_help = sub_action.help or "No description available."
            # Categorize based on whether the argument is required
            if sub_action.required:
                required_args.append(f"- `{option_strings}` (Required): {option_help}")
            else:
                optional_args.append(f"- `{option_strings}` (Optional): {option_help}")

        # Add Required arguments section
        if required_args:
            lines.extend(required_args)
            lines.append("\n")

        # Add Optional arguments section
        if optional_args:
            lines.extend(optional_args)
            lines.append("\n")

        # Add Example Usage
        lines.append(f"**Example Usage:**")
        lines.append("\n")
        lines.append("```bash")
        lines.append(f"neps {cmd} --help")
        lines.append("```")
        lines.append("\n")

    # Write the lines to the specified markdown file
    with open(filename, "w") as f:
        f.write("\n".join(lines))


def handle_report_config(args: argparse.Namespace) -> None:
    """Handles the report-config command which updates reports for
    trials in the NePS state."""
    # Load run_args from the provided path or default to run_config.yaml
    if args.run_args:
        run_args_path = Path(args.run_args)
    else:
        run_args_path = Path("run_config.yaml")
    if not run_args_path.exists():
        print(f"Error: run_args file {run_args_path} does not exist.")
        return

    with open(run_args_path, "r") as f:
        run_args = yaml.safe_load(f)

    # Get root_directory from run_args
    root_directory = run_args.get("root_directory")
    if not root_directory:
        print("Error: 'root_directory' is not specified in the run_args file.")
        return

    root_directory = Path(root_directory)
    if not root_directory.exists():
        print(f"Error: The directory {root_directory} does not exist.")
        return

    neps_state = load_neps_state(root_directory)
    if neps_state is None:
        return

    # Load the existing trial by ID
    try:
        trial = neps_state.unsafe_retry_get_trial_by_id(args.trial_id)
        if not trial:
            print(f"No trial found with ID {args.trial_id}")
            return
    except Exception as e:
        print(f"Error fetching trial with ID {args.trial_id}: {e}")
        return None

    # Update state of the trial and create report
    report = trial.set_complete(
        report_as=args.reported_as,
        time_end=args.time_end,
        objective_to_minimize=args.objective_to_minimize,
        cost=args.cost,
        learning_curve=args.learning_curve,
        err=Exception(args.err) if args.err else None,
        tb=args.tb,
        evaluation_duration=args.duration,
        extra={},
    )

    # Update NePS state
    try:
        neps_state._report_trial_evaluation(
            trial=trial, report=report, worker_id=args.worker_id
        )
    except Exception as e:
        print(f"Error updating the report for trial {args.trial_id}: {e}")
        return None

    print(f"Report for trial ID {trial.metadata.id} has been successfully updated.")

    print("\n--- Report Summary ---")
    print(f"Trial ID: {trial.metadata.id}")
    print(f"Reported As: {report.reported_as}")
    print(f"Time Ended: {convert_timestamp(trial.metadata.time_end)}")
    print(
        f"Loss: {report.objective_to_minimize if report.objective_to_minimize is not None else 'N/A'}"
    )
    print(f"Cost: {report.cost if report.cost is not None else 'N/A'}")
    print(f"Evaluation Duration: {format_duration(report.evaluation_duration)}")

    if report.learning_curve:
        print(f"Learning Curve: {' '.join(map(str, report.learning_curve))}")
    else:
        print("Learning Curve: N/A")

    if report.err:
        print(f"Error Type: {type(report.err).__name__}")
        print(f"Error Message: {str(report.err)}")
        print("Traceback:")
        print(report.tb if report.tb else "N/A")
    else:
        print("Error: None")

    print("----------------------\n")


def parse_time_end(time_str: str) -> float:
    """Parses a UNIX timestamp or a human-readable time string
    and returns a UNIX timestamp."""
    try:
        # First, try to interpret the input as a UNIX timestamp
        return float(time_str)
    except ValueError:
        pass

    try:
        # If that fails, try to interpret it as a human-readable datetime
        # string (YYYY-MM-DD HH:MM:SS)
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()  # Convert to UNIX timestamp (float)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid time format: '{time_str}'. "
            f"Use UNIX timestamp or 'YYYY-MM-DD HH:MM:SS'."
        )


def main() -> None:
    """CLI entry point.

    This function sets up the command-line interface (CLI) for NePS using argparse.
    It defines the available subcommands and their respective arguments.
    """
    raise NotImplementedError("Sorry, this is not currently implemented.")
    # > parser = argparse.ArgumentParser(description="NePS Command Line Interface")
    # > subparsers = parser.add_subparsers(
    # >     dest="command", help="Available commands: init, run"
    # > )

    # > # Subparser for "init" command
    # > parser_init = subparsers.add_parser("init", help="Generate 'run_args' YAML file")
    # > parser_init.add_argument(
    # >     "--config-path",
    # >     type=str,
    # >     default=None,
    # >     help="Optional custom path for generating the configuration file. "
    # >     "Default is 'run_config.yaml'.",
    # > )
    # > parser_init.add_argument(
    # >     "--template",
    # >     type=str,
    # >     choices=["basic", "complete"],
    # >     default="basic",
    # >     help="Optional, options between different templates. Required configs(basic) vs "
    # >     "all neps configs (complete)",
    # > )
    # > parser_init.add_argument(
    # >     "--database",
    # >     action="store_true",
    # >     help="If set, creates a NEPS state. Requires an existing config.yaml.",
    # > )
    # > parser_init.set_defaults(func=init_config)

    # > # Subparser for "run" command
    # > parser_run = subparsers.add_parser("run", help="Run a neural pipeline search.")
    # > # Adding arguments to the 'run' subparser with defaults
    # > parser_run.add_argument(
    # >     "--run",
    # >     type=str,
    # >     help="Path to the YAML configuration file(s).",
    # >     nargs="*",
    # >     default=None,
    # > )

    # > parser_run.add_argument(
    # >     "--pipeline-space",
    # >     type=str,
    # >     default=None,
    # >     help="Path to the YAML file defining the search space for the optimization. "
    # >     "This can be provided here or defined within the 'run_args' YAML file. "
    # >     "(default: %(default)s)",
    # > )
    # > parser_run.add_argument(
    # >     "--root-directory",
    # >     type=str,
    # >     default=None,
    # >     help="The directory to save progress to. This is also used to synchronize "
    # >     "multiple calls for parallelization. (default: %(default)s)",
    # > )
    # > parser_run.add_argument(
    # >     "--overwrite-working-directory",
    # >     action="store_true",
    # >     default=False,  # noqa: FBT003
    # >     help="If set, deletes the working directory at the start of the run. "
    # >     "This is useful, for example, when debugging a evaluate_pipeline function. "
    # >     "(default: %(default)s)",
    # > )
    # > # Create a mutually exclusive group for post-run summary flags
    # > summary_group = parser_run.add_mutually_exclusive_group(required=False)
    # > summary_group.add_argument(
    # >     "--post-run-summary",
    # >     action="store_true",
    # >     default=True,  # noqa: FBT003
    # >     help="Provide a summary of the results after running. (default: %(default)s)",
    # > )
    # > summary_group.add_argument(
    # >     "--no-post-run-summary",
    # >     action="store_false",
    # >     dest="post_run_summary",
    # >     help="Do not provide a summary of the results after running.",
    # > )
    # > parser_run.add_argument(
    # >     "--max-evaluations-total",
    # >     type=int,
    # >     default=None,
    # >     help="Total number of evaluations to run. (default: %(default)s)",
    # > )
    # > parser_run.add_argument(
    # >     "--max-evaluations-per-run",
    # >     type=int,
    # >     default=None,
    # >     help="Number of evaluations a specific call should maximally do. "
    # >     "(default: %(default)s)",
    # > )
    # > parser_run.add_argument(
    # >     "--continue-until-max-evaluation-completed",
    # >     action="store_true",
    # >     default=False,  # noqa: FBT003
    # >     help="If set, only stop after max-evaluations-total have been completed. This "
    # >     "is only relevant in the parallel setting. (default: %(default)s)",
    # > )
    # > parser_run.add_argument(
    # >     "--max-cost-total",
    # >     type=float,
    # >     default=None,
    # >     help="No new evaluations will start when this cost is exceeded. Requires "
    # >     "returning a cost in the evaluate_pipeline function, e.g., `return dict("
    # >     "objective_to_minimize=objective_to_minimize, cost=cost)`. (default: %(default)s)",
    # > )
    # > parser_run.add_argument(
    # >     "--ignore-errors",
    # >     action="store_true",
    # >     default=False,  # noqa: FBT003
    # >     help="If set, ignore errors during the optimization process. (default: %("
    # >     "default)s)",
    # > )
    # > parser_run.add_argument(
    # >     "--objective_to_minimize-value-on-error",
    # >     type=float,
    # >     default=None,
    # >     help="Loss value to assume on error. (default: %(default)s)",
    # > )
    # > parser_run.add_argument(
    # >     "--cost-value-on-error",
    # >     type=float,
    # >     default=None,
    # >     help="Cost value to assume on error. (default: %(default)s)",
    # > )

    # > parser_run.add_argument(
    # >     "--optimizer",
    # >     type=str,
    # >     default="default",
    # >     help="String key of optimizer algorithm to use for optimization. (default: %("
    # >     "default)s)",
    # > )

    # > parser_run.add_argument(
    # >     "--optimizer-kwargs",
    # >     type=str,
    # >     nargs="+",
    # >     help="Additional keyword arguments as key=value pairs for the optimizer.",
    # > )

    # > parser_run.set_defaults(func=run_optimization)

    # > # Subparser for "info-config" command
    # > parser_info_config = subparsers.add_parser(
    # >     "info-config", help="Provides information about specific config."
    # > )
    # > parser_info_config.add_argument(
    # >     "id", type=str, help="The configuration ID to be used."
    # > )
    # > parser_info_config.add_argument(
    # >     "--root-directory",
    # >     type=str,
    # >     help="Optional: The path to your root_directory. If not provided, "
    # >     "it will be loaded from run_config.yaml.",
    # > )
    # > parser_info_config.set_defaults(func=info_config)

    # > # Subparser for "errors" command
    # > parser_errors = subparsers.add_parser("errors", help="List all errors.")
    # > parser_errors.add_argument(
    # >     "--root-directory",
    # >     type=str,
    # >     help="Optional: The path to your "
    # >     "root_directory. If not provided, it will be "
    # >     "loaded from run_config.yaml.",
    # > )
    # > parser_errors.set_defaults(func=load_neps_errors)

    # > # Subparser for "sample-config" command
    # > parser_sample_config = subparsers.add_parser(
    # >     "sample-config", help="Sample configurations from the existing NePS state."
    # > )
    # > parser_sample_config.add_argument(
    # >     "--worker-id",
    # >     type=str,
    # >     default="cli",
    # >     help="The worker ID for which the configuration is being sampled.",
    # > )
    # > parser_sample_config.add_argument(
    # >     "--run-args",
    # >     type=str,
    # >     help="Optional: Path to the YAML configuration file.",
    # > )
    # > parser_sample_config.add_argument(
    # >     "--number-of-configs",
    # >     type=int,
    # >     default=1,
    # >     help="Optional: Number of configurations to sample (default: 1).",
    # > )
    # > parser_sample_config.set_defaults(func=sample_config)

    # > report_parser = subparsers.add_parser(
    # >     "report-config", help="Report of a specific trial"
    # > )
    # > report_parser.add_argument("trial_id", type=str, help="ID of the trial to report")
    # > report_parser.add_argument(
    # >     "reported_as",
    # >     type=str,
    # >     choices=["success", "failed", "crashed"],
    # >     help="Outcome of the trial",
    # > )
    # > report_parser.add_argument(
    # >     "--worker-id",
    # >     type=str,
    # >     default="cli",
    # >     help="The worker ID for which the configuration is being sampled.",
    # > )
    # > report_parser.add_argument(
    # >     "--objective_to_minimize", type=float, help="Loss value of the trial"
    # > )
    # > report_parser.add_argument(
    # >     "--run-args", type=str, help="Path to the YAML file containing run configurations"
    # > )
    # > report_parser.add_argument(
    # >     "--cost", type=float, help="Cost value of the trial (optional)"
    # > )
    # > report_parser.add_argument(
    # >     "--learning-curve",
    # >     type=float,
    # >     nargs="+",
    # >     help="Learning curve as a list of floats (optional), provided like this "
    # >     "--learning-curve 0.9 0.3 0.1",
    # > )
    # > report_parser.add_argument(
    # >     "--duration", type=float, help="Duration of the evaluation in sec (optional)"
    # > )
    # > report_parser.add_argument("--err", type=str, help="Error message if any (optional)")
    # > report_parser.add_argument(
    # >     "--tb", type=str, help="Traceback information if any (optional)"
    # > )
    # > report_parser.add_argument(
    # >     "--time-end",
    # >     type=parse_time_end,  # Using the custom parser function
    # >     help="The time the trial ended as either a "
    # >     "UNIX timestamp (float) or in 'YYYY-MM-DD HH:MM:SS' format",
    # > )
    # > report_parser.set_defaults(func=handle_report_config)

    # > # Subparser for "status" command
    # > parser_status = subparsers.add_parser(
    # >     "status", help="Check the status of the NePS run."
    # > )
    # > parser_status.add_argument(
    # >     "--root-directory",
    # >     type=str,
    # >     help="Optional: The path to your root_directory. If not provided, "
    # >     "it will be loaded from run_config.yaml.",
    # > )
    # > parser_status.add_argument(
    # >     "--pending", action="store_true", help="Show only pending trials."
    # > )
    # > parser_status.add_argument(
    # >     "--evaluating", action="store_true", help="Show only evaluating trials."
    # > )
    # > parser_status.add_argument(
    # >     "--succeeded", action="store_true", help="Show only succeeded trials."
    # > )
    # > parser_status.set_defaults(func=status)

    # > # Subparser for "results" command
    # > parser_results = subparsers.add_parser(
    # >     "results", help="Display results of the NePS run."
    # > )
    # > parser_results.add_argument(
    # >     "--root-directory",
    # >     type=str,
    # >     help="Optional: The path to your root_directory. If not provided, "
    # >     "it will be loaded from run_config.yaml.",
    # > )
    # > parser_results.add_argument(
    # >     "--plot", action="store_true", help="Plot the results if set."
    # > )

    # > # Create a mutually exclusive group for dump options
    # > dump_group = parser_results.add_mutually_exclusive_group()
    # > dump_group.add_argument(
    # >     "--dump-all-configs",
    # >     type=str,
    # >     choices=["csv", "json", "parquet"],
    # >     help="Dump all trials to a file in the specified format (csv, json, parquet).",
    # > )
    # > dump_group.add_argument(
    # >     "--dump-incumbents",
    # >     type=str,
    # >     choices=["csv", "json", "parquet"],
    # >     help="Dump incumbent trials to a file in the specified format "
    # >     "(csv, json, parquet).",
    # > )

    # > parser_results.set_defaults(func=results)

    # > # Subparser for "help" command
    # > parser_help = subparsers.add_parser("help", help="Displays help information.")
    # > parser_help.set_defaults(func=print_help)

    # > # updating documentation
    # > generate_markdown_from_parser(parser, "cli.md")
    # > args = parser.parse_args()

    # > if hasattr(args, "func"):
    # >     args.func(args)
    # > else:
    # >     parser.print_help()


if __name__ == "__main__":
    main()
