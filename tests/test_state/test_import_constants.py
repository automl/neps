"""Imported trials keep the constants of a classic search space."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from neps.normalization import _normalize_imported_config
from neps.space import HPOConstant, HPOFloat, HPOInteger, SearchSpace
from neps.space.neps_spaces.parameters import Fidelity, Float, Integer, PipelineSpace


def _space() -> SearchSpace:
    return SearchSpace(
        {
            "x": HPOFloat(0.0, 1.0),
            "epochs": HPOInteger(1, 10, is_fidelity=True),
            "wd": HPOConstant(1e-6),
        }
    )


def test_constants_are_kept_and_completed() -> None:
    space = _space()
    assert _normalize_imported_config(space, {"x": 0.5, "wd": 1e-6}) == {
        "x": 0.5,
        "epochs": 10,
        "wd": 1e-6,
    }
    # Missing constants are filled in, like missing fidelities
    assert _normalize_imported_config(space, {"x": 0.5}) == {
        "x": 0.5,
        "epochs": 10,
        "wd": 1e-6,
    }


def test_mismatching_constant_is_kept_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="neps.normalization"):
        config = _normalize_imported_config(_space(), {"x": 0.5, "wd": 1e-3})
    assert config["wd"] == 1e-3
    assert "wd" in caplog.text


def test_unknown_keys_are_still_dropped() -> None:
    config = _normalize_imported_config(_space(), {"x": 0.5, "unknown": 1})
    assert "unknown" not in config


class _Pipeline(PipelineSpace):
    x = Float(lower=0.0, upper=1.0)
    epochs = Fidelity(Integer(lower=1, upper=10))
    wd = 1e-6


def test_import_trials_keeps_constants(tmp_path: Path) -> None:
    from neps import import_trials
    from neps.state import NePSState
    from neps.state.pipeline_eval import UserResultDict

    root = tmp_path / "run"
    import_trials(
        [({"x": 0.5, "epochs": 10, "wd": 1e-6}, UserResultDict(objective_to_minimize=1.0))],
        root_directory=root,
        pipeline_space=_Pipeline(),
        optimizer=("bayesian_optimization", {"ignore_fidelity": True}),
    )
    (trial,) = NePSState.create_or_load(root, load_only=True).lock_and_read_trials().values()
    assert trial.config == {"x": 0.5, "epochs": 10, "wd": 1e-6}
