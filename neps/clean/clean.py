"""Functionality for cleaning failed trials from a neps working directory.

This module provides utilities to remove failed/crashed trials using the proper
NePSState API, which handles:
1. Removing trial directories from disk
2. Deleting the trial cache for regeneration
3. Updating shared_errors.jsonl
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from neps.state.neps_state import NePSState
from neps.state.trial import Trial

if TYPE_CHECKING:
    from neps.state.trial import State

logger = logging.getLogger(__name__)


def clean_trials_by_state(  # noqa: C901, PLR0912
    root_directory: Path,
    *,
    desired_states: list[State] | None = None,
    dry_run: bool = False,
) -> dict[State | str, int]:
    """Remove trials with specified states from the working directory.

    Gets all trials from NePSState and removes those matching the desired_states.
    By default, removes trials in FAILED, CRASHED, and CORRUPTED states.

    Uses the NePSState API to ensure proper handling of:
    1. Trial directory removal from disk
    2. Trial cache deletion for regeneration
    3. Shared error log updates

    Args:
        root_directory: The neps working directory path
        desired_states: List of Trial.State values to remove. If None, defaults to
            [Trial.State.FAILED, Trial.State.CRASHED, Trial.State.CORRUPTED]
        dry_run: If True, only report what would be deleted without making changes

    Returns:
        A dictionary with statistics:
        - Trial.State.FAILED: Number of failed trials removed
        - Trial.State.CRASHED: Number of crashed trials removed
        - Trial.State.CORRUPTED: Number of corrupted trials removed
        - 'errors_cleaned': Number of error entries removed from shared_errors.jsonl
        - 'total_removed': Total number of trials removed
    """
    if desired_states is None:
        desired_states = [
            Trial.State.FAILED,
            Trial.State.CRASHED,
            Trial.State.CORRUPTED,
        ]

    root_directory = Path(root_directory).resolve()
    if not root_directory.exists():
        raise FileNotFoundError(f"Working directory not found: {root_directory}")

    stats: dict[State | str, int] = {state: 0 for state in desired_states}
    stats["errors_cleaned"] = 0
    stats["total_removed"] = 0

    try:
        state = NePSState.create_or_load(root_directory, load_only=True)
    except FileNotFoundError:
        logger.warning(f"No NePSState found in {root_directory}")
        return stats

    trials = state.lock_and_read_trials()
    if not trials:
        logger.info("No trials found in the working directory")
        return stats

    trial_ids_to_remove: list[str] = []

    for trial_id, trial in trials.items():
        trial_state = trial.metadata.state

        if trial_state in desired_states:
            trial_ids_to_remove.append(trial_id)
            stats[trial_state] += 1
            stats["total_removed"] += 1

            action = "Removing" if not dry_run else "[DRY RUN] Would remove"
            logger.info(f"{action} {trial_state.value} trial: {trial_id}")

    if dry_run:
        logger.info("=" * 70)
        logger.info("DRY RUN - No changes will be made")
        logger.info("=" * 70)
        for state_to_check in desired_states:
            logger.info(f"  {state_to_check.value} trials: {stats[state_to_check]}")
        logger.info("=" * 70)

    if trial_ids_to_remove:
        original_errors = state.lock_and_get_errors()
        original_error_count = len(original_errors.errs)

        for trial_id in trial_ids_to_remove:
            if not dry_run:
                try:
                    state.lock_and_remove_trial_by_id(trial_id)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to remove trial {trial_id}: {e}")
                    continue

        if not dry_run:
            updated_errors = state.lock_and_get_errors()
            stats["errors_cleaned"] = original_error_count - len(updated_errors.errs)
            logger.info(
                f"Cleaned {stats['total_removed']} trials and "
                f"{stats['errors_cleaned']} error entries"
            )
        else:
            stats["errors_cleaned"] = sum(
                1 for err in original_errors.errs if err.trial_id in trial_ids_to_remove
            )
            logger.info(f"~{stats['errors_cleaned']} error entries would be cleaned")

    return stats


def clean_trials_by_id(
    root_directory: Path,
    trial_ids: list[str],
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Remove trials by their specific trial IDs.

    Uses the NePSState API to ensure proper handling of:
    1. Trial directory removal from disk
    2. Trial cache deletion for regeneration
    3. Shared error log updates

    Args:
        root_directory: The neps working directory path
        trial_ids: List of trial IDs to remove
        dry_run: If True, only report what would be deleted without making changes

    Returns:
        A dictionary with statistics:
        - 'removed': Number of trials successfully removed
        - 'not_found': Number of trial IDs that were not found
        - 'errors_cleaned': Number of error entries removed from shared_errors.jsonl
    """
    root_directory = Path(root_directory).resolve()
    if not root_directory.exists():
        raise FileNotFoundError(f"Working directory not found: {root_directory}")

    stats = {
        "removed": 0,
        "not_found": 0,
        "errors_cleaned": 0,
    }

    if not trial_ids:
        logger.warning("No trial IDs provided")
        return stats

    try:
        state = NePSState.create_or_load(root_directory, load_only=True)
    except FileNotFoundError:
        logger.warning(f"No NePSState found in {root_directory}")
        return stats

    trials = state.lock_and_read_trials()

    if dry_run:
        logger.info("=" * 70)
        logger.info("DRY RUN - No changes will be made")
        logger.info("=" * 70)

    original_errors = state.lock_and_get_errors()
    original_error_count = len(original_errors.errs)

    for trial_id in trial_ids:
        if trial_id not in trials:
            logger.warning(f"Trial {trial_id} not found")
            stats["not_found"] += 1
            continue

        trial = trials[trial_id]
        action = "Removing" if not dry_run else "[DRY RUN] Would remove"
        logger.info(f"{action} trial: {trial_id} (state: {trial.metadata.state.value})")

        if not dry_run:
            try:
                state.lock_and_remove_trial_by_id(trial_id)
                stats["removed"] += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to remove trial {trial_id}: {e}")
                continue

    if dry_run:
        logger.info(f"  Would remove: {len(trial_ids)} trials")
        stats["removed"] = len(trial_ids) - stats["not_found"]
        logger.info("=" * 70)
    else:
        updated_errors = state.lock_and_get_errors()
        stats["errors_cleaned"] = original_error_count - len(updated_errors.errs)
        logger.info(
            f"Cleaned {stats['removed']} trials and "
            f"{stats['errors_cleaned']} error entries"
        )

    return stats


def clean_failed_trials(
    root_directory: Path,
    *,
    desired_states: list[State] | None = None,
    trial_ids: list[str] | None = None,
    dry_run: bool = False,
) -> dict[State | str, int]:
    """Remove trials by state or by specific trial IDs.

    This is a convenience wrapper that delegates to either clean_trials_by_state()
    or clean_trials_by_id() based on the arguments provided.

    Args:
        root_directory: The neps working directory path
        desired_states: List of Trial.State values to remove.
            If provided, trial_ids is ignored.
            Defaults to [Trial.State.FAILED, Trial.State.CRASHED, Trial.State.CORRUPTED]
        trial_ids: List of specific trial IDs to remove.
            Only used if desired_states is None.
        dry_run: If True, only report what would be deleted without making changes

    Returns:
        A dictionary with statistics (format depends on which removal mode is used)
    """
    if trial_ids is not None and desired_states is None:
        # Use trial ID removal mode
        return clean_trials_by_id(root_directory, trial_ids, dry_run=dry_run)
    # Use state-based removal mode
    return clean_trials_by_state(
        root_directory,
        desired_states=desired_states,
        dry_run=dry_run,
    )
