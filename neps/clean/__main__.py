"""Cleans trials from a neps working directory.

Usage:
    # Remove all unsuccessful trials (FAILED, CRASHED, CORRUPTED)
    python -m neps.clean [-h] [--dry_run] root_directory

    # Remove specific trial IDs
    python -m neps.clean [-h] [--dry_run] --trial_ids TRIAL_ID root_directory

Positional arguments:
    root_directory  The working directory given to neps.run

Optional arguments:
    -h, --help         show this help message and exit
    --trial_ids        Remove specific trial IDs (space-separated list)
    --dry_run          Show what would be deleted without making changes

Note:
    We have to use the __main__.py construct due to the issues explained in
    https://stackoverflow.com/questions/43393764/python-3-6-project-structure-leads-to-runtimewarning

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from neps.clean.clean import clean_failed_trials
from neps.state.trial import Trial

parser = argparse.ArgumentParser(
    prog="python -m neps.clean",
    description="Cleans trials from a neps working directory",
)
parser.add_argument(
    "root_directory", type=Path, help="The working directory given to neps.run"
)
parser.add_argument(
    "--trial_ids",
    nargs="+",
    type=str,
    default=None,
    help="Remove specific trial IDs (space-separated list)",
)
parser.add_argument(
    "--dry_run",
    action="store_true",
    help="Show what would be deleted without making changes",
)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if args.trial_ids:
    stats = clean_failed_trials(
        args.root_directory,
        trial_ids=args.trial_ids,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        logger.info("=" * 70)
        logger.info("Cleaning complete!")
        logger.info("=" * 70)
        logger.info(f"Trials removed: {stats['removed']}")
        logger.info(f"Trials not found: {stats['not_found']}")
        logger.info(f"Error entries cleaned: {stats['errors_cleaned']}")
        logger.info("=" * 70)
else:
    desired_states = [
        Trial.State.FAILED,
        Trial.State.CRASHED,
        Trial.State.CORRUPTED,
    ]

    stats = clean_failed_trials(
        args.root_directory,
        desired_states=desired_states,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        logger.info("=" * 70)
        logger.info("Cleaning complete!")
        logger.info("=" * 70)
        for state in desired_states:
            logger.info(f"{state.value} trials removed: {stats[state]}")
        logger.info(f"Total trials removed: {stats['total_removed']}")
        logger.info(f"Error entries cleaned: {stats['errors_cleaned']}")
        logger.info("=" * 70)
