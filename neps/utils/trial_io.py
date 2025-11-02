"""Utility functions for loading trials from a file."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, ValuesView
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from neps.state.neps_state import TrialRepo
from neps.state.pipeline_eval import UserResultDict

if TYPE_CHECKING:
    from neps.state.trial import Trial


def load_trials_from_pickle(
    root_dir: Path | str,
) -> Sequence[tuple[Mapping[str, Any], UserResultDict]]:
    """Load trials from a pickle-based TrialRepo.

    Args:
        root_dir (Path | str): The root directory containing the 'configs' folder.

    Returns:
        Sequence[tuple[Mapping[str, Any], UserResultDict]]: A sequence of tuples,
        each containing the trial configuration and its corresponding report
        as a dictionary.
    """
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
    trials: ValuesView[Trial] = (
        TrialRepo(
            root_dir / "configs",
        )
        .latest(create_cache_if_missing=False)
        .values()
    )

    return [
        (trial.config, cast(UserResultDict, asdict(trial.report)))
        for trial in trials
        if trial.report is not None
    ]
