"""Tests for the scaling plots in `summary/pre_train/` and `summary/downstream/`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import portalocker as pl
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import extrapolaw
from extrapolaw.plot import scaling_plots
from extrapolaw.space.extrapolaw_spaces.parameters import Float, Integer, PipelineSpace
from extrapolaw.state.extrapolaw_state import ExtrapolawState


class _Space(PipelineSpace):
    width = Integer(8, 512, log=True, is_compute_scaling=True)
    lr = Float(1e-4, 1e-1, log=True)


def _scaling(width: int, lr: float) -> dict:
    n_params = width**2
    return {
        "objective_to_minimize": 1.0 + 10.0 / n_params**0.3 + abs(np.log10(lr) + 2.5),
        "cost": 6.0 * n_params * 1e6,
        "inference_cost": 2.0 * n_params,
        "downstream_performances": {"task_acc": 0.9 - 1.0 / width},
        "n_params": n_params,
        "n_data": 1e6,
        "info_dict": {"dropout": 0.0 if width < 64 else 0.1},
    }


def _zero_cost(width: int, lr: float) -> dict:
    return {"objective_to_minimize": lr, "cost": 0.0}


def _run(root: Path, evaluate_pipeline: object) -> None:
    extrapolaw.acquire(
        evaluate_pipeline=evaluate_pipeline,  # type: ignore[arg-type]
        pipeline_space=_Space(),
        root_directory=root,
        optimizer="random_search",
        worker_evaluations_to_spend=6,
    )


def test_scaling_plots_are_written(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _run(root, _scaling)
    summary = root / "summary"
    assert not (summary / "pre_train").exists()

    extrapolaw.analyze(root)
    for name in (
        "pre_train/loss_over_cost",
        "pre_train/loss_over_inference_cost",
        "pre_train/parameters/width_over_cost",
        "pre_train/parameters/lr_over_cost",
        "pre_train/parameters/n_params_over_cost",
        "pre_train/parameters/width_over_inference_cost",
        "downstream/task_acc/task_acc_over_cost",
        "downstream/task_acc/task_acc_over_inference_cost",
        "downstream/task_acc/parameters/width_over_cost",
    ):
        assert (summary / f"{name}.png").exists()
        assert len(pd.read_csv(summary / f"{name}.csv")) == 6

    assert not (summary / "pre_train/parameters/inference_cost_over_cost.png").exists()
    downstream = pd.read_csv(summary / "downstream/task_acc/task_acc_over_cost.csv")
    assert (downstream["downstream/task_acc"] < 0.5).all()

    table = pd.read_csv(summary / "pre_train/loss_over_cost.csv")
    assert table.loc[table["loss"].idxmin(), "on_frontier"]
    pareto = pd.read_csv(summary / "pre_train/pareto_over_cost.csv")
    assert len(pareto) == table["on_frontier"].sum()
    assert {"trial_id", "cost", "loss", "n_params", "width", "lr"} <= set(pareto)
    assert "width" in (summary / "pre_train/pareto_over_cost.txt").read_text()

    predictions = pd.read_csv(summary / "pre_train/predictions_over_cost.csv")
    largest = table["cost"].max()
    expected = [10 * largest, 100 * largest]
    assert list(predictions["target_cost"]) == pytest.approx(expected)
    assert {"loss", "n_params", "n_data", "width", "lr"} <= set(predictions)
    accuracy = pd.read_csv(summary / "downstream/task_acc/predictions_over_cost.csv")
    assert (accuracy["task_acc"] > 0.5).all()


def _events(log_dir: Path) -> EventAccumulator:
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0, "tensors": 0})
    events.Reload()
    return events


def _incumbent(events: EventAccumulator) -> list:

    scalars = events.Scalars("scaling/incumbent/loss")
    return sorted(scalars, key=lambda event: event.step)


def _incumbent_steps(log_dir: Path) -> list[int]:
    return [event.step for event in _incumbent(_events(log_dir))]


def _frontiers(log_dir: Path) -> dict[int, str]:
    """The frontier of loss over cost at each step."""
    texts = _events(log_dir).Tensors("scaling/frontier/loss_over_cost/text_summary")
    return {event.step: event.tensor_proto.string_val[0].decode() for event in texts}


def test_tensorboard_steps_stay_the_same(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _run(root, _scaling)
    state = ExtrapolawState.create_or_load(path=root, load_only=True)
    trials = list(state._trial_repo.get_valid_evaluated_trials().values())
    first = min(trials, key=lambda trial: trial.metadata.time_end or 0.0)
    log_dir = tmp_path / "tensorboard"

    scaling_plots.log_to_tensorboard([t for t in trials if t is not first], log_dir)
    scaling_plots.log_to_tensorboard(trials, log_dir)

    assert _incumbent_steps(log_dir) == [1, 2, 3, 4, 5, 6]
    logged = (log_dir / "logged_trial_ids.txt").read_text().split()
    assert logged[-1] == first.id

    frontiers = _frontiers(log_dir)
    assert sorted(frontiers) == [1, 2, 3, 4, 5, 6]
    assert frontiers[1].startswith(f"({logged[0]}, ")
    assert frontiers[1].count("(") == 1


class _BusyLocker:
    """A TensorBoard lock that another worker holds for longer than the timeout."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def lock(self) -> NoReturn:
        raise pl.exceptions.LockException


def test_tensorboard_is_fail_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "results"
    _run(root, _scaling)
    state = ExtrapolawState.create_or_load(path=root, load_only=True)
    trials = list(state._trial_repo.get_valid_evaluated_trials().values())
    log_dir = tmp_path / "tensorboard"
    with monkeypatch.context() as patch, caplog.at_level(logging.WARNING):
        patch.setattr(scaling_plots, "FileLocker", _BusyLocker)
        scaling_plots.log_to_tensorboard(trials, log_dir)
    assert "TensorBoard is busy" in caplog.text

    # With the lock free again, the missed runs are logged, each with its frontier
    scaling_plots.log_to_tensorboard(trials, log_dir)
    assert _incumbent_steps(log_dir) == [1, 2, 3, 4, 5, 6]
    assert sorted(_frontiers(log_dir)) == [1, 2, 3, 4, 5, 6]


def test_no_scaling_plots_without_a_positive_cost(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _run(root, _zero_cost)
    extrapolaw.analyze(root)

    assert (root / "summary" / "full.csv").exists()
    assert not (root / "summary" / "pre_train").exists()


def test_predict_saves_the_targets_and_the_convergence(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _run(root, _scaling)

    state = ExtrapolawState.create_or_load(path=root, load_only=True)
    assert state.lock_and_get_optimizer_info()["derived"]["predictor"] == "power_law"

    extrapolaw.predict(root, [1e14, 1e15], target_inference_costs=[1e6])
    summary = root / "summary"
    targets = json.loads((summary / "targets.json").read_text())
    assert targets == {"cost": [1e14, 1e15], "inference_cost": [1e6]}
    inference = pd.read_csv(summary / "pre_train/predictions_over_inference_cost.csv")
    assert list(inference["target_inference_cost"]) == [1e6]
    assert list(inference["n_data"]) == [1e6]
    for name in ("loss_over_cost", "loss_over_inference_cost", "task_acc_over_cost"):
        assert (summary / "convergence" / f"{name}.png").exists()
    assert len(pd.read_csv(summary / "convergence/loss_over_cost.csv")) > 0

    extrapolaw.analyze(root)
    predictions = pd.read_csv(summary / "pre_train/predictions_over_cost.csv")
    assert list(predictions["target_cost"]) == [1e14, 1e15]
