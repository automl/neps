# flake8: noqa
"""This module provides a command-line interface (CLI) for NePS."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional
import neps
from neps.api import Default
from neps.utils.run_args import load_and_return_object
from neps.state.filebased import (
    create_or_load_filebased_neps_state,
    load_filebased_neps_state,
)
from neps.exceptions import VersionedResourceDoesNotExistsError, TrialNotFoundError


def init_config(args: argparse.Namespace) -> None:
    """Creates a 'run_args' configuration YAML file template if it does not already
    exist.
    """
    config_path = Path(args.config_path) if args.config_path else Path("config.yaml")
    if not config_path.exists():
        with config_path.open("w") as file:
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
    else:
        pass


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
        run_pipeline = load_and_return_object(module_path, function_name, "run_pipeline")

    else:
        run_pipeline = args.run_pipeline

    kwargs = {}
    if args.searcher_kwargs:
        kwargs = parse_kv_pairs(args.searcher_kwargs)  # convert kwargs

    # Collect arguments from args and prepare them for neps.run
    options = {
        "run_args": args.run_args,
        "run_pipeline": run_pipeline,
        "pipeline_space": args.pipeline_space,
        "root_directory": args.root_directory,
        "overwrite_working_directory": args.overwrite_working_directory,
        "post_run_summary": args.post_run_summary,
        "development_stage_id": args.development_stage_id,
        "task_id": args.task_id,
        "max_evaluations_total": args.max_evaluations_total,
        "max_evaluations_per_run": args.max_evaluations_per_run,
        "continue_until_max_evaluation_completed": (
            args.continue_until_max_evaluation_completed
        ),
        "max_cost_total": args.max_cost_total,
        "ignore_errors": args.ignore_errors,
        "loss_value_on_error": args.loss_value_on_error,
        "cost_value_on_error": args.cost_value_on_error,
        "searcher": args.searcher,
        **kwargs,
    }
    logging.basicConfig(level=logging.INFO)
    neps.run(**options)


def info_config(args: argparse.Namespace) -> None:
    """Handles the info-config command by providing information based on directory
    and id."""
    directory_path = Path(args.directory)
    config_id = args.id

    if not directory_path.exists() or not directory_path.is_dir():
        print(
            f"Error: The directory {directory_path} does not exist or is not a "
            f"directory."
        )
        return
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
    print(f"  Time Sampled: {trial.metadata.time_sampled}")
    print(f"  Evaluating Worker ID: {trial.metadata.evaluating_worker_id}")
    print(f"  Evaluation Duration: {trial.metadata.evaluation_duration}")
    print(f"  Time Started: {trial.metadata.time_started}")
    print(f"  Time End: {trial.metadata.time_end}")

    if trial.report is not None:
        print("\nReport:")
        print(f"  Loss: {trial.report.loss}")
        print(f"  Cost: {trial.report.cost}")
        print(f"  Reported As: {trial.report.reported_as}")
    else:
        print("No report available.")


def load_neps_errors(args: argparse.Namespace) -> None:
    """Handles the 'errors' command by loading errors from the neps_state."""
    directory_path = Path(args.directory)

    if not directory_path.exists() or not directory_path.is_dir():
        print(
            f"Error: The directory {directory_path} does not exist or is not a "
            f"directory."
        )
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


def print_help(args: Optional[argparse.Namespace] = None) -> None:
    """Prints help information for the NEPS CLI."""
    help_text = """
Usage: neps [COMMAND] [OPTIONS]

Available Commands:
-------------------

neps init --config-path </path/to/config.yaml>
    Generates a 'run_args' YAML template file.
    Optional custom path for generating the configuration file.
    Example:
    ```bash
    neps init --config-path /path/to/config.yaml
    ```

neps run [OPTIONS]
    Runs a neural pipeline search.
    Options:
    --run-args <path_to_run_args> (Path to the YAML configuration file.)
    --run-pipeline <path_to_module:function_name> (Optional: Python file and
    function name.)
    --pipeline-space <path_to_yaml> (Path to the YAML file defining the search space.)
    --root-directory <path> (The directory to save progress.)
    --overwrite-working-directory (If set, deletes the working directory at the start
    of the run.)
    --development-stage-id <id> (Identifier for the current development stage.)
    --task-id <id> (Identifier for the current task.)
    --post-run-summary/--no-post-run-summary (Whether to provide a summary after running.)
    --max-evaluations-total <int> (Total number of evaluations to run.)
    --max-evaluations-per-run <int> (Number of evaluations a specific call should
    maximally do.)
    --continue-until-max-evaluation-completed (If set, only stop after
    max-evaluations-total have been completed.)
    --max-cost-total <float> (No new evaluations will start when this cost is exceeded.)
    --ignore-errors (If set, ignore errors during the optimization process.)
    --loss-value-on-error <float> (Loss value to assume on error.)
    --cost-value-on-error <float> (Cost value to assume on error.)
    --searcher <key> (String key of searcher algorithm to use for optimization.)
    --searcher-kwargs <key=value>... (Additional keyword arguments for the searcher.)
    Example:
    ```bash
    neps run --run-args /path/to/config.yaml
    ```

neps info-config <id> <directory>
    Provides detailed information about a specific configuration by its ID.
    Example:
    ```bash
    neps info-config 17 /path/to/directory
    ```

neps errors <directory>
    Lists all errors found in the specified directory's NEPS state.
    Example:
    ```bash
    neps errors /path/to/directory
    ```

neps help
    Displays this help message.
    Example:
    ```bash
    neps help
    ```
    """
    print(help_text)


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
        "Default is 'config.yaml'.",
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
        "directory", type=str, help="The path to your root_directory."
    )
    parser_info_config.set_defaults(func=info_config)

    # Subparser for "errors" command
    parser_errors = subparsers.add_parser("errors", help="List all errors.")
    parser_errors.add_argument(
        "directory", type=str, help="The path to your root_directory."
    )
    parser_errors.set_defaults(func=load_neps_errors)

    # Subparser for "help" command
    parser_help = subparsers.add_parser("help", help="Displays help information.")
    parser_help.set_defaults(func=print_help)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
