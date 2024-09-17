# flake8: noqa
"""This module provides a command-line interface (CLI) for NePS."""

from __future__ import annotations
import warnings
from datetime import timedelta, datetime
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Suppress specific warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
from neps.state.trial import Trial
import argparse
import logging
import yaml
from pathlib import Path
from typing import Optional, List
import neps
from neps.api import Default
from neps.utils.run_args import *
from neps.optimizers.base_optimizer import BaseOptimizer
from neps.utils.run_args import load_and_return_object
from neps.state.filebased import (
    create_or_load_filebased_neps_state,
    load_filebased_neps_state,
)
from neps.exceptions import VersionedResourceDoesNotExistsError, TrialNotFoundError
from neps.status.status import get_summary_dict
from neps.api import _run_args


def get_root_directory(args: argparse.Namespace) -> Path | None:
    """Load the root directory from the provided argument or from the config.yaml file."""
    if args.root_directory:
        return Path(args.root_directory)

    config_path = Path("run_config.yaml")
    if config_path.exists():
        with config_path.open("r") as file:
            config = yaml.safe_load(file)
        root_directory = config.get(ROOT_DIRECTORY)
        if root_directory:
            root_directory_path = Path(root_directory)
            if not root_directory_path.exists() or not root_directory_path.is_dir():
                print(
                    f"Error: The directory {root_directory_path} does not exist or is "
                    f"not a "
                    f"directory."
                )
                return None
            return root_directory_path
        else:
            raise ValueError(
                "The config.yaml file exists but does not contain 'root_directory'."
            )
    else:
        raise ValueError(
            "Either the root_directory must be provided as an argument or config.yaml "
            "must exist with a 'root_directory' key."
        )


def init_config(args: argparse.Namespace) -> None:
    """Creates a 'run_args' configuration YAML file template if it does not already
    exist.
    """
    config_path = Path(args.config_path) if args.config_path else Path("run_config.yaml")
    if not config_path.exists():
        with config_path.open("w") as file:
            template = args.template if args.template else "basic"
            if template == "basic":
                file.write(
                    """# Add your NEPS configuration settings here

run_pipeline:
  path: "path/to/your/run_pipeline.py"
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

run_pipeline:
  path: path/to/your/run_pipeline.py  # Path to the function file
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
development_stage_id:
task_id:

# Parallelization Setup
max_evaluations_per_run:
continue_until_max_evaluation_completed: true

# Error Handling
loss_value_on_error:
cost_value_on_error:
ignore_errors:

# Customization Options
searcher: hyperband       # Internal key to select a NePS optimizer.

# Hooks
pre_load_hooks:
"""
                )
    elif args.state_machine:
        pass
        # create_or_load_filebased_neps_state()
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
    if not isinstance(args.run_pipeline, Default):
        module_path, function_name = args.run_pipeline.split(":")
        run_pipeline = load_and_return_object(module_path, function_name, RUN_PIPELINE)

    else:
        run_pipeline = args.run_pipeline

    kwargs = {}
    if args.searcher_kwargs:
        kwargs = parse_kv_pairs(args.searcher_kwargs)  # convert kwargs

    # Collect arguments from args and prepare them for neps.run
    options = {
        RUN_ARGS: args.run_args,
        RUN_PIPELINE: run_pipeline,
        PIPELINE_SPACE: args.pipeline_space,
        ROOT_DIRECTORY: args.root_directory,
        OVERWRITE_WORKING_DIRECTORY: args.overwrite_working_directory,
        POST_RUN_SUMMARY: args.post_run_summary,
        DEVELOPMENT_STAGE_ID: args.development_stage_id,
        TASK_ID: args.task_id,
        MAX_EVALUATIONS_TOTAL: args.max_evaluations_total,
        MAX_EVALUATIONS_PER_RUN: args.max_evaluations_per_run,
        CONTINUE_UNTIL_MAX_EVALUATION_COMPLETED: (
            args.continue_until_max_evaluation_completed
        ),
        MAX_COST_TOTAL: args.max_cost_total,
        IGNORE_ERROR: args.ignore_errors,
        LOSS_VALUE_ON_ERROR: args.loss_value_on_error,
        COST_VALUE_ON_ERROR: args.cost_value_on_error,
        SEARCHER: args.searcher,
        **kwargs,
    }
    logging.basicConfig(level=logging.INFO)
    neps.run(**options)


def info_config(args: argparse.Namespace) -> None:
    """Handles the info-config command by providing information based on directory
    and id."""
    directory_path = get_root_directory(args)
    if directory_path is None:
        return
    config_id = args.id

    try:
        neps_state = load_filebased_neps_state(directory_path)
    except VersionedResourceDoesNotExistsError:
        print(f"No NePS state found in the directory {directory_path}.")
        return
    try:
        trial = neps_state.get_trial_by_id(config_id)
    except TrialNotFoundError:
        print(f"No trial found with ID {config_id}.")
        return

    print("Trial Information:")
    print(f"  Trial ID: {trial.metadata.id}")
    print(f"  State: {trial.state}")
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
        print(f"  Loss: {trial.report.loss}")
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

    try:
        neps_state = load_filebased_neps_state(directory_path)
    except VersionedResourceDoesNotExistsError:
        print(f"No NePS state found in the directory {directory_path}.")
        return
    errors = neps_state.get_errors()

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

    # Load the YAML configuration
    with run_args_path.open("r") as f:
        run_args = get_run_args_from_yaml(run_args_path)

    # Get root_directory from the run_args
    root_directory = run_args.get(ROOT_DIRECTORY)
    if not root_directory:
        print("Error: 'root_directory' is not specified in the run_args file.")
        return

    root_directory = Path(root_directory)
    if not root_directory.exists():
        print(f"Error: The directory {root_directory} does not exist.")
        return

    # Load the NePS state from the root_directory
    try:
        neps_state = load_filebased_neps_state(root_directory)
    except VersionedResourceDoesNotExistsError:
        print(f"No NePS state found in the directory {root_directory}.")
        return

    # Get the worker_id and number_of_configs from arguments
    worker_id = args.worker_id
    num_configs = args.number_of_configs if args.number_of_configs else 1

    # Create the optimizer
    try:
        searcher_info = {
            "searcher_name": "",
            "searcher_alg": "",
            "searcher_selection": "",
            "neps_decision_tree": True,
            "searcher_args": {},
        }

        # Call _run_args() to create the optimizer
        optimizer, _ = _run_args(
            searcher_info=searcher_info,
            pipeline_space=run_args.get(PIPELINE_SPACE),
            max_cost_total=run_args.get(MAX_COST_TOTAL, None),
            ignore_errors=run_args.get(IGNORE_ERROR, False),
            loss_value_on_error=run_args.get(LOSS_VALUE_ON_ERROR, None),
            cost_value_on_error=run_args.get(COST_VALUE_ON_ERROR, None),
            searcher=run_args.get(SEARCHER, "default"),
            **run_args.get(SEARCHER_KWARGS, {}),
        )
    except Exception as e:
        print(f"Error creating optimizer: {e}")
        return

    # Sample trials
    for _ in range(num_configs):
        try:
            trial = neps_state.sample_trial(optimizer, worker_id=worker_id)
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
    directory_path = get_root_directory(args)
    if directory_path is None:
        return

    try:
        neps_state = load_filebased_neps_state(directory_path)
    except VersionedResourceDoesNotExistsError:
        print(f"No NePS state found in the directory {directory_path}.")
        return

    summary = get_summary_dict(directory_path, add_details=True)

    # Calculate the number of trials in different states
    evaluating_trials_count = sum(
        1
        for trial in neps_state.get_all_trials().values()
        if trial.state.name == "EVALUATING"
    )
    pending_trials_count = summary["num_pending_configs"]
    succeeded_trials_count = summary["num_evaluated_configs"] - summary["num_error"]
    failed_trials_count = summary["num_error"]
    pending_with_worker_count = summary["num_pending_configs_with_worker"]

    # Print summary
    print("NePS Status:")
    print("-----------------------------")
    print(f"Optimizer: {neps_state.optimizer_info().info['searcher_alg']}")
    print(f"Succeeded Trials: {succeeded_trials_count}")
    print(f"Failed Trials (Errors): {failed_trials_count}")
    print(f"Active Trials: {evaluating_trials_count}")
    print(f"Pending Trials: {pending_trials_count}")
    print(f"Pending Trials with Worker: {pending_with_worker_count}")
    print(f"Best Loss Achieved: {summary['best_loss']}")

    print("\nLatest Trials:")
    print("-----------------------------")

    # Retrieve and sort the trials by time_sampled
    all_trials = neps_state.get_all_trials()
    sorted_trials = sorted(
        all_trials.values(), key=lambda t: t.metadata.time_sampled, reverse=True
    )

    # Filter trials based on state
    if args.pending:
        filtered_trials = [
            trial for trial in sorted_trials if trial.state.name == "PENDING"
        ]
    elif args.evaluating:
        filtered_trials = [
            trial for trial in sorted_trials if trial.state.name == "EVALUATING"
        ]
    elif args.succeeded:
        filtered_trials = [
            trial for trial in sorted_trials if trial.state.name == "SUCCESS"
        ]
    else:
        filtered_trials = sorted_trials[:7]

    # Define column headers with fixed width
    header_format = "{:<20} {:<10} {:<10} {:<40} {:<12} {:<10}"
    row_format = "{:<20} {:<10} {:<10} {:<40} {:<12} {:<10}"

    # Print the header
    print(
        header_format.format(
            "Sampled Time", "Duration", "Trial ID", "Worker ID", "State", "Loss"
        )
    )

    # Print the details of the filtered trials
    for trial in filtered_trials:
        time_sampled = convert_timestamp(trial.metadata.time_sampled)
        if trial.state.name in ["PENDING", "EVALUATING"]:
            duration = compute_duration(trial.metadata.time_sampled)
        else:
            duration = (
                format_duration(trial.metadata.evaluation_duration)
                if trial.metadata.evaluation_duration
                else "N/A"
            )
        trial_id = trial.id
        worker_id = trial.metadata.sampling_worker_id
        state = trial.state.name
        loss = (
            f"{trial.report.loss:.6f}"
            if (trial.report and trial.report.loss is not None)
            else "N/A"
        )

        print(row_format.format(time_sampled, duration, trial_id, worker_id, state, loss))

    # If no specific filter is applied, print the best trial and optimizer info
    if not args.pending and not args.evaluating and not args.succeeded:
        if summary["best_config_id"] is not None:
            print("\nBest Trial:")
            print("-----------------------------")
            print(f"ID: {summary['best_config_id']}")
            print(f"Loss: {summary['best_loss']}")
            print("Config:")
            for key, value in summary["best_config"].items():
                print(f"  {key}: {value}")

            print(
                f"\nMore details: neps info-config {summary['best_config_id']} "
                f"(use --root-directory if not using run_config.yaml)"
            )
        else:
            print("\nBest Trial:")
            print("-----------------------------")
            print("\nNo successful trial found.")

        # Display optimizer information
        optimizer_info = neps_state.optimizer_info().info
        searcher_name = optimizer_info.get("searcher_name", "N/A")
        searcher_alg = optimizer_info.get("searcher_alg", "N/A")
        searcher_args = optimizer_info.get("searcher_args", {})

        print("\nOptimizer Information:")
        print("-----------------------------")
        print(f"Name: {searcher_name}")
        print(f"Algorithm: {searcher_alg}")
        print("Parameter:")
        for arg, value in searcher_args.items():
            print(f"  {arg}: {value}")

        print("-----------------------------")


def plot_incumbents(
    all_trials: List[Trial], incumbents: List[Trial], directory_path: Path
) -> str:
    """Plot the evolution of incumbent trials over the total number of trials."""
    id_to_index = {trial.id: idx + 1 for idx, trial in enumerate(all_trials)}

    # Collect data for plotting
    x_values = [id_to_index[incumbent.id] for incumbent in incumbents]
    y_values = [
        incumbent.report.loss
        for incumbent in incumbents
        if incumbent.report is not None and incumbent.report.loss is not None
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


def results(args: argparse.Namespace) -> None:
    """Handles the 'results' command by displaying incumbents in
    reverse order and
    optionally plotting and saving the results."""
    directory_path = get_root_directory(args)
    if directory_path is None:
        return

    try:
        neps_state = load_filebased_neps_state(directory_path)
    except VersionedResourceDoesNotExistsError:
        print(f"No NePS state found in the directory {directory_path}.")
        return

    trials = neps_state.get_all_trials()
    # Sort trials by trial ID
    sorted_trials = sorted(trials.values(), key=lambda x: int(x.id))

    # Compute incumbents
    best_loss = float("inf")
    incumbents = []
    for trial in sorted_trials:
        if trial.report and trial.report.loss < best_loss:
            best_loss = trial.report.loss
            incumbents.append(trial)

    # Reverse the list for displaying, so the most recent incumbent is shown first
    incumbents_display = incumbents[::-1]

    if not args.plot:
        print(f"Results for NePS run: {directory_path}")
        print("--------------------")
        print("All Incumbent Trials:")
        header = f"{'ID':<6} {'Loss':<12} {'Config':<60}"
        print(header)
        print("-" * len(header))
        if len(incumbents_display) > 0:
            for trial in incumbents_display:
                if trial.report is not None and trial.report.loss is not None:
                    config = ", ".join(f"{k}: {v}" for k, v in trial.config.items())
                    print(f"{trial.id:<6} {trial.report.loss:<12.6f} {config:<60}")
                else:
                    print(f"Trial {trial.id} has no valid loss.")
        else:
            print("No Incumbent Trials found.")
    else:
        plot_path = plot_incumbents(sorted_trials, incumbents, directory_path)
        print(f"Plot saved to {plot_path}")


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
    --state-machine (Optional: Creates a NEPS state. Requires an existing config.yaml.)

neps run [OPTIONS]
    Runs a neural pipeline search.
    Options:
    --run-args <path_to_run_args> (Path to the YAML configuration file.)
    --run-pipeline <path_to_module:function_name> (Path and function for the pipeline.)
    --pipeline-space <path_to_yaml> (Path to the YAML defining the search space.)
    --root-directory <path> (Optional: Directory for saving progress and
    synchronization. Default is 'root_directory' from run_config.yaml if not provided.)
    --overwrite-working-directory (Deletes the working directory at the start of the run.)
    --development-stage-id <id> (Identifier for the development stage.)
    --task-id <id> (Identifier for the task.)
    --post-run-summary/--no-post-run-summary (Toggle summary after running.)
    --max-evaluations-total <int> (Total number of evaluations to run.)
    --max-evaluations-per-run <int> (Max evaluations per run call.)
    --continue-until-max-evaluation-completed (Continue until max evaluations are completed.)
    --max-cost-total <float> (Max cost before halting new evaluations.)
    --ignore-errors (Ignore errors during optimization.)
    --loss-value-on-error <float> (Assumed loss value on error.)
    --cost-value-on-error <float> (Assumed cost value on error.)
    --searcher <key> (Searcher algorithm key for optimization.)
    --searcher-kwargs <key=value>... (Additional kwargs for the searcher.)

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


def main() -> None:
    """CLI entry point.

    This function sets up the command-line interface (CLI) for NePS using argparse.
    It defines the available subcommands and their respective arguments.

    Available commands:
        - init: Generates a 'run_args' YAML template file.
        - run: Runs the optimization with specified configuration.
    """
    parser = argparse.ArgumentParser(description="NePS Command Line Interface")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands: init, run"
    )

    # Subparser for "init" command
    parser_init = subparsers.add_parser("init", help="Generate 'run_args' " "YAML file")
    parser_init.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional custom path for generating the configuration file. "
        "Default is 'run_config.yaml'.",
    )
    parser_init.add_argument(
        "--template",
        type=str,
        choices=["basic", "complete"],
        default="basic",
        help="Optional, options between different templates. Required configs(basic) vs "
        "all neps configs (complete)",
    )
    parser_init.add_argument(
        "--state-machine",
        action="store_true",
        help="If set, creates a NEPS state. Requires an existing config.yaml.",
    )
    parser_init.set_defaults(func=init_config)

    # Subparser for "run" command
    parser_run = subparsers.add_parser("run", help="Run a neural pipeline search.")
    # Adding arguments to the 'run' subparser with defaults
    parser_run.add_argument(
        "--run-args",
        type=str,
        help="Path to the YAML configuration file.",
        default=Default(None),
    )
    parser_run.add_argument(
        "--run-pipeline",
        type=str,
        help="Optional: Provide the path to a Python file and a function name separated "
        "by a colon, e.g., 'path/to/module.py:function_name'. "
        "If provided, it overrides the run_pipeline setting from the YAML "
        "configuration.",
        default=Default(None),
    )

    parser_run.add_argument(
        "--pipeline-space",
        type=str,
        default=Default(None),
        help="Path to the YAML file defining the search space for the optimization. "
        "This can be provided here or defined within the 'run_args' YAML file. "
        "(default: %(default)s)",
    )
    parser_run.add_argument(
        "--root-directory",
        type=str,
        default=Default(None),
        help="The directory to save progress to. This is also used to synchronize "
        "multiple calls for parallelization. (default: %(default)s)",
    )
    parser_run.add_argument(
        "--overwrite-working-directory",
        action="store_true",
        default=Default(False),  # noqa: FBT003
        help="If set, deletes the working directory at the start of the run. "
        "This is useful, for example, when debugging a run_pipeline function. "
        "(default: %(default)s)",
    )
    parser_run.add_argument(
        "--development-stage-id",
        type=str,
        default=Default(None),
        help="Identifier for the current development stage, used in multi-stage "
        "projects. (default: %(default)s)",
    )
    parser_run.add_argument(
        "--task-id",
        type=str,
        default=Default(None),
        help="Identifier for the current task, useful in projects with multiple tasks. "
        "(default: %(default)s)",
    )
    # Create a mutually exclusive group for post-run summary flags
    summary_group = parser_run.add_mutually_exclusive_group(required=False)
    summary_group.add_argument(
        "--post-run-summary",
        action="store_true",
        default=Default(True),  # noqa: FBT003
        help="Provide a summary of the results after running. (default: %(default)s)",
    )
    summary_group.add_argument(
        "--no-post-run-summary",
        action="store_false",
        dest="post_run_summary",
        help="Do not provide a summary of the results after running.",
    )
    parser_run.add_argument(
        "--max-evaluations-total",
        type=int,
        default=Default(None),
        help="Total number of evaluations to run. (default: %(default)s)",
    )
    parser_run.add_argument(
        "--max-evaluations-per-run",
        type=int,
        default=Default(None),
        help="Number of evaluations a specific call should maximally do. "
        "(default: %(default)s)",
    )
    parser_run.add_argument(
        "--continue-until-max-evaluation-completed",
        action="store_true",
        default=Default(False),  # noqa: FBT003
        help="If set, only stop after max-evaluations-total have been completed. This "
        "is only relevant in the parallel setting. (default: %(default)s)",
    )
    parser_run.add_argument(
        "--max-cost-total",
        type=float,
        default=Default(None),
        help="No new evaluations will start when this cost is exceeded. Requires "
        "returning a cost in the run_pipeline function, e.g., `return dict("
        "loss=loss, cost=cost)`. (default: %(default)s)",
    )
    parser_run.add_argument(
        "--ignore-errors",
        action="store_true",
        default=Default(False),  # noqa: FBT003
        help="If set, ignore errors during the optimization process. (default: %("
        "default)s)",
    )
    parser_run.add_argument(
        "--loss-value-on-error",
        type=float,
        default=Default(None),
        help="Loss value to assume on error. (default: %(default)s)",
    )
    parser_run.add_argument(
        "--cost-value-on-error",
        type=float,
        default=Default(None),
        help="Cost value to assume on error. (default: %(default)s)",
    )

    parser_run.add_argument(
        "--searcher",
        type=str,
        default=Default("default"),
        help="String key of searcher algorithm to use for optimization. (default: %("
        "default)s)",
    )

    parser_run.add_argument(
        "--searcher-kwargs",
        type=str,
        nargs="+",
        help="Additional keyword arguments as key=value pairs for the searcher.",
    )

    parser_run.set_defaults(func=run_optimization)

    # Subparser for "info-config" command
    parser_info_config = subparsers.add_parser(
        "info-config", help="Provides information about " "specific config."
    )
    parser_info_config.add_argument(
        "id", type=str, help="The configuration ID to be used."
    )
    parser_info_config.add_argument(
        "--root-directory",
        type=str,
        help="Optional: The path to your root_directory. If not provided, "
        "it will be loaded from run_config.yaml.",
    )
    parser_info_config.set_defaults(func=info_config)

    # Subparser for "errors" command
    parser_errors = subparsers.add_parser("errors", help="List all errors.")
    parser_errors.add_argument(
        "--root-directory",
        type=str,
        help="Optional: The path to your "
        "root_directory. If not provided, it will be "
        "loaded from run_config.yaml.",
    )
    parser_errors.set_defaults(func=load_neps_errors)

    # Subparser for "sample-config" command
    parser_sample_config = subparsers.add_parser(
        "sample-config", help="Sample configurations from the existing NePS state."
    )
    parser_sample_config.add_argument(
        "worker_id",
        type=str,
        help="The worker ID for which the configuration is being sampled.",
    )
    parser_sample_config.add_argument(
        "--run-args",
        type=str,
        help="Optional: Path to the YAML configuration file.",
    )
    parser_sample_config.add_argument(
        "--number-of-configs",
        type=int,
        default=1,
        help="Optional: Number of configurations to sample (default: 1).",
    )
    parser_sample_config.set_defaults(func=sample_config)

    # Subparser for "status" command
    parser_status = subparsers.add_parser(
        "status", help="Check the status of the NePS run."
    )
    parser_status.add_argument(
        "--root-directory",
        type=str,
        help="Optional: The path to your root_directory. If not provided, "
        "it will be loaded from run_config.yaml.",
    )
    parser_status.add_argument(
        "--pending", action="store_true", help="Show only pending trials."
    )
    parser_status.add_argument(
        "--evaluating", action="store_true", help="Show only evaluating trials."
    )
    parser_status.add_argument(
        "--succeeded", action="store_true", help="Show only succeeded trials."
    )
    parser_status.set_defaults(func=status)

    # Subparser for "results" command
    parser_results = subparsers.add_parser(
        "results", help="Display results of the NePS run."
    )
    parser_results.add_argument(
        "--root-directory",
        type=str,
        help="Optional: The path to your root_directory. If not provided, "
        "it will be loaded from run_config.yaml.",
    )
    parser_results.add_argument(
        "--plot", action="store_true", help="Plot the results if set."
    )
    parser_results.set_defaults(func=results)

    # Subparser for "help" command
    parser_help = subparsers.add_parser("help", help="Displays help information.")
    parser_help.set_defaults(func=print_help)

    # updating documentation
    generate_markdown_from_parser(parser, "cli.md")
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
