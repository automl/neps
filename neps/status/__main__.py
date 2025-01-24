"""Displays status information about a working directory of a neps.run.

Usage:
    python -m neps.status [-h] [--best_objective_to_minimizees] [--best_configs]
        [--all_configs] working_directory

Positional arguments:
    working_directory  The working directory given to neps.run

Optional arguments:
    -h, --help         show this help message and exit
    --best_objective_to_minimizees      Show the trajectory of the best
        objective_to_minimize across evaluations
    --best_configs     Show the trajectory of the best configs and their
        objective_to_minimizees
                     across evaluations
    --all_configs      Show all configs and their objective_to_minimizees

Note:
    We have to use the __main__.py construct due to the issues explained in
    https://stackoverflow.com/questions/43393764/python-3-6-project-structure-leads-to-runtimewarning

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .status import status

# fmt: off
parser = argparse.ArgumentParser(
    prog="python -m neps.status",
    description="Displays status information about a working directory of a neps.run",
)
parser.add_argument("root_directory", type=Path,
                    help="The working directory given to neps.run")
args = parser.parse_args()

logging.basicConfig(level=logging.WARN)
status(args.root_directory, print_summary=True)
