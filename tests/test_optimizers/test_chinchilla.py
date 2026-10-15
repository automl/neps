from __future__ import annotations

import time

import matplotlib as mpl
import pytest

from autoscan.optimizers import load_optimizer
from autoscan.optimizers.algorithms import chinchilla
from autoscan.optimizers.chinchilla import (
    ChinchillaGuidedScaling,
    _pareto_front,
    _ScalingPoint,
)
from autoscan.optimizers.optimizer import SampledConfig, ScalingOptimizer
from autoscan.space import (
    HPOCategorical,
    HPOConstant,
    HPOFloat,
    HPOInteger,
    SearchSpace,
)
from autoscan.state import BudgetInfo, Trial
from autoscan.utils.cost_estimator import CostEstimator

mpl.use("Agg")

MAX_COST_PER_CONFIG = 1e10
BUDGET = BudgetInfo(total_cost_to_spend=1e11)


def _space() -> SearchSpace:
    return SearchSpace(
        {
            "width": HPOInteger(8, 256, is_compute_scaling=True),
            "depth": HPOInteger(1, 8, is_compute_scaling=True),
            "n_data": HPOInteger(1_000, 50_000, is_compute_scaling=True),
            "lr": HPOFloat(1e-5, 1e-1, log=True),
            "opt": HPOCategorical(["adam", "sgd"]),
            "dataset": HPOConstant("toy"),
        }
    )


def _n_params(width: int, depth: int) -> int:
    return width * width * depth


def _flops(width: int, depth: int, n_data: int) -> float:
    return 6.0 * _n_params(width, depth) * n_data


def _optimizer(**kwargs) -> ChinchillaGuidedScaling:
    optimizer, _ = load_optimizer(
        (
            "chinchilla",
            {
                "strategy": "approach2",
                "data_parameter_name": "n_data",
                "params_estimator": _n_params,
                **kwargs,
            },
        ),
        _space(),
        cost_estimator=CostEstimator.from_callable(_flops),
        max_cost_per_config=MAX_COST_PER_CONFIG,
    )
    assert isinstance(optimizer, ChinchillaGuidedScaling)
    return optimizer


def _trial(sampled: SampledConfig, objective: float, **report) -> Trial:
    trial = Trial.new(
        trial_id=sampled.id,
        config=sampled.config,
        location="",
        previous_trial=None,
        previous_trial_location=None,
        time_sampled=time.time(),
        worker_id="0",
    )
    trial.report = trial.set_complete(
        report_as="success",
        time_end=time.time(),
        objective_to_minimize=objective,
        err=None,
        tb=None,
        evaluation_duration=0.0,
        **{"cost": None, "learning_curve": None, "extra": None, **report},
    )
    return trial


def _run(optimizer: ChinchillaGuidedScaling, n_trials: int) -> dict[str, Trial]:
    trials: dict[str, Trial] = {}
    for _ in range(n_trials):
        sampled = optimizer(trials, BUDGET)
        assert isinstance(sampled, SampledConfig)
        trials[sampled.id] = _trial(sampled, objective=_flops_of(sampled.config) ** -0.3)
    return trials


def _flops_of(config) -> float:
    return _flops(config["width"], config["depth"], config["n_data"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "approach1", "data_multipliers": [1, 5, 20, 80]},
        {"strategy": "approach2"},
    ],
)
def test_laid_out_configs_respect_bounds_and_budget(kwargs) -> None:
    optimizer = _optimizer(**kwargs)
    assert isinstance(optimizer, ScalingOptimizer)
    assert optimizer.base_optimizer is None  # Laid out on the first ask

    sampled = optimizer({}, BUDGET)
    assert isinstance(sampled, SampledConfig)
    assert sampled.id == "0"

    configs = optimizer.config_list
    assert len(configs) > 1
    assert sum(_flops_of(c) for c in configs) <= BUDGET.total_cost_to_spend
    for config in configs:
        assert set(config) == {"width", "depth", "n_data", "lr", "opt", "dataset"}
        assert 8 <= config["width"] <= 256
        assert 1 <= config["depth"] <= 8
        assert 1_000 <= config["n_data"] <= 50_000
        assert _flops_of(config) <= MAX_COST_PER_CONFIG
    # Everything that is not scaled is drawn once and held fixed
    assert len({(c["lr"], c["opt"], c["dataset"]) for c in configs}) == 1
    assert configs[0]["opt"] in ("adam", "sgd")
    assert configs[0]["dataset"] == "toy"
    assert len({c["width"] * c["depth"] for c in configs}) > 1


def test_configs_are_not_repeated_across_asks() -> None:
    trials = _run(_optimizer(), n_trials=6)
    assert list(trials) == [str(i) for i in range(6)]
    assert len({tuple(sorted(t.config.items())) for t in trials.values()}) == 6


def test_worker_budget_is_used_without_a_total_one() -> None:
    optimizer = _optimizer()
    optimizer({}, BudgetInfo(worker_cost_to_spend=1e10))
    assert sum(_flops_of(c) for c in optimizer.config_list) <= 1e10


def test_missing_requirements_raise() -> None:
    with pytest.raises(ValueError, match="needs a cost budget"):
        _optimizer()({}, None)
    with pytest.raises(ValueError, match="needs a `cost_estimator`"):
        chinchilla(
            _space(),
            strategy="approach2",
            data_parameter_name="n_data",
            params_estimator=_n_params,
        )({}, BUDGET)
    with pytest.raises(ValueError, match="needs `data_multipliers`"):
        _optimizer(strategy="approach1")
    with pytest.raises(ValueError, match="`data_parameter_name` must be one of"):
        _optimizer(data_parameter_name="tokens")
    with pytest.raises(ValueError, match="does not support fidelities"):
        chinchilla(
            SearchSpace(
                {
                    "width": HPOInteger(8, 256, is_compute_scaling=True),
                    "n_data": HPOInteger(1_000, 50_000),
                    "epochs": HPOInteger(1, 9, is_fidelity=True),
                }
            ),
            strategy="approach2",
            data_parameter_name="n_data",
            params_estimator=_n_params,
        )
    with pytest.raises(ValueError, match="at least one parameter"):
        chinchilla(
            SearchSpace({"width": HPOInteger(8, 256), "n_data": HPOInteger(1, 9)}),
            strategy="approach2",
            data_parameter_name="n_data",
            params_estimator=_n_params,
        )


def test_artifacts_do_not_rely_on_the_reported_extra() -> None:
    optimizer = _optimizer()
    trials = _run(optimizer, n_trials=6)
    assert optimizer.get_trial_artifacts(dict(list(trials.items())[:4])) == []

    artifacts = optimizer.get_trial_artifacts(trials)
    assert [a.name for a in artifacts] == ["flops_vs_params", "flops_vs_data"]

    # Flops, parameter count and data are derived from the config, not from `extra`
    misleading = {
        trial.id: _trial(
            SampledConfig(trial.id, trial.config),
            objective=trial.report.objective_to_minimize,
            extra={"flops": -1, "n_param": -1, "n_data": -1},
        )
        for trial in trials.values()
    }
    points = optimizer._scaling_points(misleading, lc_aware=False)
    assert points == optimizer._scaling_points(trials, lc_aware=False)
    for point, trial in zip(points, trials.values(), strict=True):
        config = trial.config
        assert point.n_params == _n_params(config["width"], config["depth"])
        assert point.n_data == config["n_data"]
        assert point.flops == _flops_of(config)


def test_front_keeps_parameter_count_monotone_in_flops() -> None:
    # The second point is on the pareto front, but has more parameters than the third
    points = [
        _ScalingPoint(n_params=10, n_data=1, loss=0.5, flops=100.0),
        _ScalingPoint(n_params=30, n_data=1, loss=0.4, flops=200.0),
        _ScalingPoint(n_params=20, n_data=1, loss=0.3, flops=300.0),
        _ScalingPoint(n_params=20, n_data=1, loss=0.9, flops=400.0),  # Dominated
    ]
    assert _pareto_front(points) == [points[0], points[2]]
    assert _pareto_front([]) == []


def test_learning_curves_are_used_by_approach1() -> None:
    optimizer = _optimizer(strategy="approach1", data_multipliers=[1, 5, 20, 80])
    trials: dict[str, Trial] = {}
    for _ in range(5):
        sampled = optimizer(trials, BUDGET)
        flops, n_data = _flops_of(sampled.config), sampled.config["n_data"]
        curve = [[n_data * f, flops * f, (flops * f) ** -0.3] for f in (0.25, 0.5, 1.0)]
        trials[sampled.id] = _trial(
            sampled, objective=curve[-1][-1], learning_curve=curve
        )

    assert len(optimizer._scaling_points(trials, lc_aware=True)) == 15
    assert [a.name for a in optimizer.get_trial_artifacts(trials)] == [
        "flops_vs_params",
        "flops_vs_data",
        "flops_vs_loss_with_lc",
    ]
