""" Plot incumbent from a root directory of a neps.run

Usage:
    python -m neps.plot [-h] [--scientific_mode] [--key_to_extract] [--benchmarks]
    [--algorithms] [--consider_continuations] [--n_workers] [--x_range] [--log_x]
    [--log_y] [--filename] [--extension] [--dpi]
                          root_directory

Positional arguments:
    root_directory              The root directory given to neps.run

Optional arguments:
    -h, --help                  Show this help message and exit
    --scientific_mode           If true, plot from a tree-structured root_directory: benchmark={}/algorithm={}/seed={}
    --key_to_extract            The metric to be used on the x-axis (if active, make sure run_pipeline returns the metric in the info_dict)
    --benchmarks                List of benchmarks to plot
    --algorithms                List of algorithms to plot
    --consider_continuations    If true, toggle calculation of continuation costs
    --n_workers                 Number of parallel processes of neps.run
    --x_range                   Bound x-axis (e.g. 1 10)
    --log_x                     If true, toggle logarithmic scale on the x-axis
    --log_y                     If true, toggle logarithmic scale on the y-axis
    --filename                  Filename
    --extension                 Image format
    --dpi                       Image resolution


Note:
    We have to use the __main__.py construct due to the issues explained in
    https://stackoverflow.com/questions/43393764/python-3-6-project-structure-leads-to-runtimewarning

"""

import argparse
import logging
from pathlib import Path

from .plot import plot

# fmt: off
parser = argparse.ArgumentParser(
    prog="python -m neps.plot",
    description="Plot incumbent from a root directory of a neps.run"
)
parser.add_argument(
    "root_directory",
    type=Path,
    help="The root directory given to neps.run"
)
parser.add_argument(
    "--scientific_mode",
    action="store_true",
    help="If true, plot from a tree-structured root_directory"
)
parser.add_argument(
    "--key_to_extract",
    help="The metric to be used on the x-axis (if "
         "active, make sure run_pipeline returns "
         "the metric in the info_dict)")
parser.add_argument(
    "--benchmarks",
    default=["example"],
    nargs="+",
    help="List of benchmarks to plot",
)
parser.add_argument(
    "--algorithms",
    default=["neps"],
    nargs="+",
    help="List of algorithms to plot",
)
parser.add_argument(
    "--consider_continuations",
    action="store_true",
    help="If true, toggle calculation of continuation costs"
)
parser.add_argument(
    "--n_workers",
    type=int,
    default=1,
    help="Number of parallel processes of neps.run",
)
parser.add_argument(
    "--x_range",
    nargs='+',
    type=float,
    help="Bound x-axis (e.g. 1 10)"
)
parser.add_argument(
    "--log_x",
    action="store_true",
    help="If true, toggle logarithmic scale on the x-axis"
)
parser.add_argument(
    "--log_y",
    action="store_true",
    help="If true, toggle logarithmic scale on the y-axis"
)
parser.add_argument(
    "--filename",
    default="incumbent_trajectory",
    help="Filename",
)
parser.add_argument(
    "--extension",
    default="png",
    choices=["png", "pdf"],
    help="Image format",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=100,
    help="Image resolution",
)

args = parser.parse_args()
# fmt: on

logging.basicConfig(level=logging.WARN)
if args.x_range is not None and len(args.x_range) == 2:
    args.x_range = tuple(args.x_range)
plot(
    root_directory=args.root_directory,
    scientific_mode=args.scientific_mode,
    key_to_extract=args.key_to_extract,
    benchmarks=args.benchmarks,
    algorithms=args.algorithms,
    consider_continuations=args.consider_continuations,
    n_workers=args.n_workers,
    x_range=args.x_range,
    log_x=args.log_x,
    log_y=args.log_y,
    filename=args.filename,
    extension=args.extension,
    dpi=args.dpi,
)
