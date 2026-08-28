"""Cleans trials from a neps working directory.

Usage:
    # Reset all unsuccessful trials (FAILED, CRASHED, CORRUPTED) to pending
    python -m neps.clean [-h] [--root-dir ROOT_DIR] [--dry-run] [--delete]

    # Reset specific trial IDs
    python -m neps.clean [-h] [--root-dir ROOT_DIR] [--dry-run] [--delete] --trial-ids TRIAL_ID

Optional arguments:
    -h, --help           show this help message and exit
    --root-dir ROOT_DIR  The working directory given to neps.run
                         (default: neps_results)
    --trial-ids          Only clean specific trial IDs (space-separated list)
    --dry-run            Show what would change without making changes
    --delete             Delete matched trials entirely instead of the default
                         behavior of resetting them to pending (keeping their
                         config, clearing their report) for re-evaluation

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
from neps.status.status import post_run_csv

parser = argparse.ArgumentParser(
    prog="python -m neps.clean",
    description="Cleans trials from a neps working directory",
)
parser.add_argument(
    "--root-dir",
    dest="root_directory",
    type=Path,
    default=Path("neps_results"),
    help="The working directory given to neps.run (default: neps_results)",
)
parser.add_argument(
    "--trial-ids",
    dest="trial_ids",
    nargs="+",
    type=str,
    default=None,
    help="Only clean specific trial IDs (space-separated list)",
)
parser.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    help="Show what would change without making changes",
)
parser.add_argument(
    "--delete",
    dest="delete",
    action="store_true",
    help=(
        "Delete matched trials entirely instead of the default behavior of "
        "resetting them to pending (keeping their config, clearing their report) "
        "for re-evaluation"
    ),
)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

action_verb = "deleted" if args.delete else "reset"

if args.trial_ids:
    stats = clean_failed_trials(
        args.root_directory,
        trial_ids=args.trial_ids,
        dry_run=args.dry_run,
        delete=args.delete,
    )

    if not args.dry_run:
        logger.info("=" * 70)
        logger.info("Cleaning complete!")
        logger.info("=" * 70)
        logger.info(f"Trials {action_verb}: {stats['removed']}")
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
        delete=args.delete,
    )

    if not args.dry_run:
        logger.info("=" * 70)
        logger.info("Cleaning complete!")
        logger.info("=" * 70)
        for state in desired_states:
            logger.info(f"{state.value} trials {action_verb}: {stats[state]}")
        logger.info(f"Total trials {action_verb}: {stats['total_removed']}")
        logger.info(f"Error entries cleaned: {stats['errors_cleaned']}")
        logger.info("=" * 70)
        post_run_csv(args.root_directory)
