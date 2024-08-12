"""This module provides a command-line interface (CLI) for NePS."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import neps
from neps.api import Default
from neps.utils.run_args import load_and_return_object


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
    parser_init = subparsers.add_parser("init", help="Generate 'run_args' YAML file")
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

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
