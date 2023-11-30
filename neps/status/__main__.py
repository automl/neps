""" Displays status information about a working directory of a neps.run

Usage:
    python -m neps.status [-h] [--best_losses] [--best_configs] [--all_configs]
                          working_directory

Positional arguments:
    working_directory  The working directory given to neps.run

Optional arguments:
    -h, --help         show this help message and exit
    --best_losses      Show the trajectory of the best loss across evaluations
    --best_configs     Show the trajectory of the best configs and their losses
                     across evaluations
    --all_configs      Show all configs and their losses

Note:
    We have to use the __main__.py construct due to the issues explained in
    https://stackoverflow.com/questions/43393764/python-3-6-project-structure-leads-to-runtimewarning

"""

import argparse
import logging
from pathlib import Path

from .status import status

# fmt: off
parser = argparse.ArgumentParser(
    prog="python -m neps.status",
    description="Displays status information about a working directory of a neps.run"
)
parser.add_argument("root_directory", type=Path,
                    help="The working directory given to neps.run")
parser.add_argument("--best_losses", action="store_true",
                    help="Show the trajectory of the best loss across evaluations")
parser.add_argument("--best_configs", action="store_true",
                    help="Show the trajectory of the best configs and their losses across evaluations")
parser.add_argument("--all_configs", action="store_true",
                    help="Show all configs and their losses")
args = parser.parse_args()
# fmt: on

logging.basicConfig(level=logging.WARN)
status(args.root_directory, args.best_losses, args.best_configs, args.all_configs)
