from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("carbs")

from autoscan.optimizers import load_optimizer
from autoscan.optimizers.algorithms import carbs
from autoscan.optimizers.optimizer import SampledConfig, ScalingOptimizer
from autoscan.space import (
    HPOCategorical,
    HPOConstant,
    HPOFloat,
    HPOInteger,
    SearchSpace,
)
from autoscan.state import Trial
from autoscan.utils.cost_estimator import CostEstimator


def _space() -> SearchSpace:
    return SearchSpace(
        {
            "width": HPOInteger(16, 1024, log=True, is_compute_scaling=True),
            "lr": HPOFloat(1e-5, 1e-1, log=True, prior=1e-3),
            "dropout": HPOFloat(0.0, 1.0),
            "dataset": HPOConstant("toy"),
        }
    )


def _trial(sampled: SampledConfig, objective: float | None, cost: float | None) -> Trial:
    trial = Trial.new(
        trial_id=sampled.id,
        config=sampled.config,
        location="",
        previous_trial=None,
        previous_trial_location=None,
        time_sampled=time.time(),
        worker_id="0",
    )
    if objective is not None:
        trial.report = trial.set_complete(
            report_as="success",
            time_end=time.time(),
            objective_to_minimize=objective,
            cost=cost,
            learning_curve=None,
            err=None,
            tb=None,
            extra=None,
            evaluation_duration=0.0,
        )
    return trial


def test_suggestions_are_complete_configs_within_bounds() -> None:
    optimizer = carbs(_space())
    sampled = optimizer({}, None)
    assert isinstance(sampled, SampledConfig)
    assert sampled.id == "0"
    assert set(sampled.config) == {"width", "lr", "dropout", "dataset"}
    assert sampled.config["dataset"] == "toy"
    assert isinstance(sampled.config["width"], int)
    assert 16 <= sampled.config["width"] <= 1024
    assert 1e-5 <= sampled.config["lr"] <= 1e-1


def test_batch_gets_consecutive_ids() -> None:
    batch = carbs(_space())({}, None, n=3)
    assert [s.id for s in batch] == ["0", "1", "2"]


def test_unsupported_spaces_are_rejected() -> None:
    with pytest.raises(NotImplementedError, match="'opt' is a HPOCategorical"):
        carbs(SearchSpace({"opt": HPOCategorical(["adam", "sgd"])}))
    with pytest.raises(ValueError, match="does not support fidelities"):
        carbs(SearchSpace({"x": HPOFloat(0, 1), "e": HPOInteger(1, 9, is_fidelity=True)}))


def test_state_is_rebuilt_from_the_trials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    optimizer = carbs(_space(), num_random_samples=2)

    trials: dict[str, Trial] = {}
    for _ in range(3):
        sampled = optimizer(trials, None)
        cost = float(sampled.config["width"])
        trials[sampled.id] = _trial(sampled, objective=1.0 / cost, cost=cost)
    pending = optimizer(trials, None)
    trials[pending.id] = _trial(pending, objective=None, cost=None)

    assert list(trials) == ["0", "1", "2", "3"]
    assert len({tuple(sorted(t.config.items())) for t in trials.values()}) == 4

    # A new optimizer, e.g. of another worker, ends up with the same CARBS state
    replayed = carbs(_space(), num_random_samples=2)._replay(trials)
    assert len(replayed.success_observations) == 3
    assert list(replayed.outstanding_suggestions) == ["3"]
    # CARBS itself writes nothing, the trials are the only record
    assert list(tmp_path.iterdir()) == []


def test_cost_comes_from_the_cost_estimator_if_not_reported() -> None:
    estimator = CostEstimator.from_callable(float)
    optimizer, info = load_optimizer(
        "carbs", _space(), cost_estimator=estimator, max_cost_per_config=100.0
    )
    assert isinstance(optimizer, ScalingOptimizer)
    assert optimizer.max_cost_per_config == 100.0
    assert info["derived"]["max_cost_per_config"] == 100.0
    assert "cost_estimator" not in info["info"]

    cheap = SampledConfig(
        "0", {"width": 64, "lr": 1e-3, "dropout": 0.5, "dataset": "toy"}
    )
    expensive = SampledConfig("1", {**cheap.config, "width": 512})
    trials = {
        "0": _trial(cheap, objective=1.0, cost=None),
        "1": _trial(expensive, objective=0.5, cost=None),
    }
    replayed = optimizer._replay(trials)
    assert [o.cost for o in replayed.success_observations] == [64.0]
    # Above `max_cost_per_config`, counted as a failure
    assert len(replayed.failure_observations) == 1


def test_missing_cost_is_an_error() -> None:
    optimizer = carbs(_space())
    sampled = optimizer({}, None)
    with pytest.raises(ValueError, match="needs the cost of every evaluation"):
        optimizer({"0": _trial(sampled, objective=1.0, cost=None)}, None)
