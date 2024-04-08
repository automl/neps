import argparse
import os

import neps


def init_config():
    # Define the paths for the configuration files
    config_path = "config.yaml"
    search_space_path = "search_space.yaml"

    # Check if 'config.yaml' exists to avoid overwriting
    if not os.path.exists(config_path):
        with open(config_path, "w") as file:
            file.write("# Add your NEPS configuration settings here\n")
        print(f"Generated template: {config_path}")
    else:
        print(f"{config_path} already exists. Skipping to avoid overwriting.")

    # Check if 'search_space.yaml' exists to avoid overwriting
    if not os.path.exists(search_space_path):
        with open(search_space_path, "w") as file:
            file.write(
                """pipeline_space:
  # Define your search space parameters here
  # Example:
  # learning_rate:
  #   type: float
  #   lower: 1e-4
  #   upper: 1e-1
  #   log: true
"""
            )
        print(f"Generated template: {search_space_path}")
    else:
        print(f"{search_space_path} already exists. Skipping to avoid overwriting.")


def run_optimization(args):
    config_path = args.config if args.config else "config.yaml"

    # Check if the config file exists
    if not os.path.isfile(config_path):
        print(f"No configuration file found at '{config_path}'.")
        print("Please create one using 'neps init' or specify the path using '--config'.")
        return

    print(f"Running optimization using configuration from {config_path}")
    neps.run(run_args=config_path)


def main():
    parser = argparse.ArgumentParser(description="NePS Command Line Interface")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands: init, run"
    )

    # Subparser for "init" command
    parser_init = subparsers.add_parser(
        "init", help="Generate starter configuration YAML files"
    )
    parser_init.set_defaults(func=init_config)

    # Subparser for "run" command
    parser_run = subparsers.add_parser(
        "run", help="Run optimization with specified configuration"
    )
    parser_run.add_argument(
        "--config", type=str, help="Path to the configuration YAML file."
    )
    parser_run.set_defaults(func=run_optimization)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
