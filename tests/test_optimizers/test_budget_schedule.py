from __future__ import annotations

import logging
import math
import time

import pytest
import torch
from botorch.models import SingleTaskGP
from gpytorch.means import ZeroMean

from scalefm.exceptions import BudgetExhaustedError
from scalefm.optimizers import bayesian_optimization as bo_module
from scalefm.optimizers.acquisition import information_theoretic
from scalefm.optimizers.acquisition.information_theoretic import (
    CostWeightedModifiedJES,
)
from scalefm.optimizers.algorithms import bayesian_optimization
from scalefm.optimizers.models.gp import LogCost, PowerLawMean
from scalefm.optimizers.optimizer import SampledConfig
from scalefm.space import HPOFloat, HPOInteger, SearchSpace
from scalefm.space.encoding import ConfigEncoder
from scalefm.state import BudgetInfo, Trial


def _space(lower: int = 1, upper: int = 1000) -> SearchSpace:
    return SearchSpace(
        {
            "n_params": HPOInteger(lower, upper, log=True),
            "n_data": HPOInteger(lower, upper, log=True),
            "x": HPOFloat(0.0, 1.0),
        }
    )


def _cost(n_params: float, n_data: float, **_: float) -> float:
    return n_params * n_data


def _objective(config: dict) -> float:
    return 1 + 10 / _cost(**config) ** 0.3 + (config["x"] - 0.3) ** 2


def _trial(sampled: SampledConfig) -> Trial:
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
        objective_to_minimize=_objective(sampled.config),
        cost=_cost(**sampled.config),
        learning_curve=None,
        err=None,
        tb=None,
        extra=None,
        evaluation_duration=0.0,
    )
    return trial


def _optimizer(schedule: str, *, max_cost: float | None = 1e6, **kwargs):
    optimizer = bayesian_optimization(
        kwargs.pop("space", _space()),
        initial_design_size=4,
        num_restarts=2,
        n_initial_start_points=32,
        budget_schedule=schedule,
        **kwargs,
    )
    optimizer.set_cost_estimator(_cost, max_cost_per_config=max_cost)
    return optimizer


def test_space_expansion_levels_and_caps() -> None:
    # Budget of 8 max cost configs: L = 3 levels, the initial design gets 1/8 of it
    optimizer = _optimizer("space_expansion")
    total = 8e6
    assert [optimizer._expansion_level(s, total) for s in (0, 1e6, 3e6, 7e6, 1e7)] == [
        0,
        1,
        2,
        3,
        3,
    ]

    def cap(spent: float, **kwargs) -> float:
        return optimizer._schedule_cap(spent, total, **kwargs)

    assert cap(0, is_initial_design=True) == 1e6 / 4
    assert cap(5e5, is_initial_design=False) == 5e5
    # Level 1 and 2 allow 2e6 and 4e6, clamped at `max_cost_per_config`
    assert cap(1e6, is_initial_design=False) == 1e6
    assert cap(3e6, is_initial_design=False) == 1e6

    # The target FLOPs of the IT-SCAN-JES acquisition does not stand in for it
    optimizer.max_cost_per_config = None
    optimizer.target_cost = 1e6
    with pytest.raises(ValueError, match="max_cost_per_config"):
        cap(1e6, is_initial_design=False)


def test_caps_are_clamped_at_the_budget_left() -> None:
    optimizer = _optimizer("space_expansion")
    assert optimizer._schedule_cap(7.5e6, 8e6, is_initial_design=False) == 5e5


def _spent(*configs: tuple[int, int]) -> dict[str, Trial]:
    sampled = [
        SampledConfig(id=str(i), config={"n_params": p, "n_data": d, "x": 0.5})
        for i, (p, d) in enumerate(configs)
    ]
    return {s.id: _trial(s) for s in sampled}


def test_too_little_budget_left_stops_the_schedule() -> None:
    optimizer = _optimizer("space_expansion")
    # 1e5 left, less than the 8e6 / 8 / 4 = 2.5e5 of an initial design config
    trials = _spent(*[(1000, 1000)] * 7, (900, 1000))
    with pytest.raises(BudgetExhaustedError, match="cheapest config"):
        optimizer(trials, BudgetInfo(worker_cost_to_spend=8e6))


def test_no_feasible_config_in_the_budget_left_stops_the_schedule() -> None:
    # 111 left, more than the 3200 / 8 / 4 = 100 of an initial design config, but all
    # configs cost at least 121
    optimizer = _optimizer("space_expansion", max_cost=400, space=_space(11, 100))
    trials = _spent((20, 100), (33, 33))
    with pytest.raises(BudgetExhaustedError, match="No feasible config"):
        optimizer(trials, BudgetInfo(worker_cost_to_spend=3200))


def test_left_budget_contraction_caps() -> None:
    optimizer = _optimizer("left_budget_contraction", max_cost=None)
    assert optimizer._schedule_cap(0, 8e6, is_initial_design=True) == 8e6 / 40
    assert optimizer._schedule_cap(3e6, 8e6, is_initial_design=False) == 5e6


@pytest.mark.parametrize("schedule", ["space_expansion", "left_budget_contraction"])
def test_sampled_configs_stay_under_the_schedule_cap(schedule: str) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(schedule)
    budget_info = BudgetInfo(worker_cost_to_spend=8e6)
    trials: dict[str, Trial] = {}
    for _ in range(6):
        spent = sum(_cost(**t.config) for t in trials.values())
        cap = optimizer._schedule_cap(
            spent, 8e6, is_initial_design=len(trials) < optimizer.n_initial_design
        )
        sampled = optimizer(trials, budget_info)
        assert _cost(**sampled.config) <= cap
        trials[sampled.id] = _trial(sampled)


def test_infeasible_level_moves_to_the_next() -> None:
    torch.manual_seed(0)
    # The initial design configs may cost 400 / 4 = 100, but all cost at least 121
    space = _space(lower=11, upper=100)
    optimizer = _optimizer("space_expansion", max_cost=400, space=space)
    sampled = optimizer({}, BudgetInfo(worker_cost_to_spend=3200))
    # Level 1 allows 3200 / 4 = 800, clamped at 400
    assert 121 <= _cost(**sampled.config) <= 400


def test_unknown_schedule_is_rejected() -> None:
    # It would otherwise mix the two schedules and fail mid-run
    with pytest.raises(ValueError, match="Unknown budget_schedule 'left_budget'"):
        _optimizer("left_budget")


def test_schedule_needs_a_budget() -> None:
    optimizer = _optimizer("space_expansion")
    with pytest.raises(ValueError, match="cost budget"):
        optimizer({}, BudgetInfo())


def _run_modified_jes(schedule: str, n_trials: int = 5) -> None:
    torch.manual_seed(0)
    # The target 1e5 lies 10x beyond the max cost per config
    optimizer = _optimizer(
        schedule, max_cost=1e4, acquisition_func_type="IT-SCAN-JES", target_cost=1e5
    )
    budget_info = BudgetInfo(worker_cost_to_spend=1e5)
    trials: dict[str, Trial] = {}
    for _ in range(n_trials):
        sampled = optimizer(trials, budget_info)
        trials[sampled.id] = _trial(sampled)


@pytest.mark.parametrize("schedule", ["space_expansion", "left_budget_contraction"])
def test_modified_jes_samples_the_optima_at_the_target_beyond_the_cap(
    schedule: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_sets: list[torch.Tensor] = []
    sample_optima = bo_module._sample_optima

    def spy(gp, candidate_set, num_optima):
        candidate_sets.append(candidate_set)
        return sample_optima(gp, candidate_set, num_optima)

    monkeypatch.setattr(bo_module, "_sample_optima", spy)
    with caplog.at_level(logging.WARNING):
        _run_modified_jes(schedule)

    assert candidate_sets
    log_cost = LogCost(ConfigEncoder.from_parameters(_space().searchables), _cost)
    for candidate_set in candidate_sets:
        distance = (log_cost.evaluate(candidate_set) - 5.0).abs()
        assert (distance < math.log10(1.5)).all()
    assert "target FLOPs" not in caplog.text


@pytest.mark.parametrize("schedule", ["space_expansion", "left_budget_contraction"])
def test_modified_jes_cost_term_spans_the_cheapest_config_to_the_max_cost(
    schedule: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bounds: list[tuple[float, float]] = []

    class Spy(CostWeightedModifiedJES):
        def __init__(self, *args, c_min_log: float, c_max_log: float, **kwargs):
            bounds.append((c_min_log, c_max_log))
            super().__init__(*args, c_min_log=c_min_log, c_max_log=c_max_log, **kwargs)

    monkeypatch.setattr(information_theoretic, "CostWeightedModifiedJES", Spy)
    _run_modified_jes(schedule)

    # The cheapest config costs 1 * 1, not the cap of the initial design configs
    assert bounds
    assert all(b == pytest.approx((0.0, 4.0)) for b in bounds)


@pytest.mark.parametrize("max_cost", [1e3, None])
def test_constraint_margin_stays_below_both_caps(max_cost: float | None) -> None:
    # The cap of the round, or else `max_cost_per_config` = 1e4, applies
    optimizer = _optimizer("left_budget_contraction", max_cost=1e4)
    cap = max_cost or 1e4
    exact = optimizer.get_constraint_func(max_cost)
    margined = optimizer.get_constraint_func(max_cost, margin=1e-3)
    on_cap = {"n_params": 100, "n_data": cap / 100, "x": 0.5}
    below_cap = {"n_params": 100, "n_data": 0.999 * cap / 100, "x": 0.5}
    assert exact(on_cap) == pytest.approx(0.0)
    assert margined(on_cap) == pytest.approx(-1e-3)
    assert margined(below_cap) == pytest.approx(0.0, abs=1e-12)


def test_acquisition_is_optimized_a_margin_below_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acq_constraints: list = []
    fit_and_acquire = bo_module.fit_and_acquire_from_gp

    def spy(*args, acq_options, **kwargs):
        acq_constraints.append(acq_options["nonlinear_inequality_constraints"][0][0])
        return fit_and_acquire(*args, acq_options=acq_options, **kwargs)

    monkeypatch.setattr(bo_module, "fit_and_acquire_from_gp", spy)
    torch.manual_seed(0)
    optimizer = _optimizer(
        "left_budget_contraction",
        max_cost=1e4,
        acquisition_func_type="IT-SCAN-JES",
        target_cost=1e5,
    )
    # A budget so large that `max_cost_per_config` is the cap of every round
    budget_info = BudgetInfo(worker_cost_to_spend=1e7)
    trials: dict[str, Trial] = {}
    for _ in range(6):
        sampled = optimizer(trials, budget_info)
        trials[sampled.id] = _trial(sampled)

    # The acquisition's constraint is the cap 1e4 tightened by the margin
    assert acq_constraints
    encoder = ConfigEncoder.from_parameters(_space().searchables)
    x = torch.rand(64, 3, dtype=torch.float64)
    costs = torch.tensor([_cost(**config) for config in encoder.decode(x)])
    expected = 1.0 - costs / 1e4 - bo_module.ACQUISITION_COST_MARGIN
    for constraint in acq_constraints:
        assert torch.allclose(constraint(x).reshape(-1), expected.to(torch.float64))


def test_modified_jes_divides_by_the_normalized_exact_cost() -> None:
    torch.manual_seed(0)
    encoder = ConfigEncoder.from_parameters(_space().searchables)
    x = torch.rand(8, 3, dtype=torch.float64)
    y = torch.rand(8, 1, dtype=torch.float64)
    gp = SingleTaskGP(x, y)
    gp.posterior(x)  # Fantasizing needs the caches of a prediction
    optima = {"optimal_inputs": x[:2], "optimal_outputs": y[:2] - 1}
    log_cost = LogCost(encoder, _cost)
    weighted = CostWeightedModifiedJES(
        gp, **optima, log_cost=log_cost, c_min_log=0.0, c_max_log=6.0
    )
    unweighted = CostWeightedModifiedJES(
        gp, **optima, log_cost=None, c_min_log=0.0, c_max_log=6.0
    )

    X = torch.rand(5, 1, 3, dtype=torch.float64)
    c_norm = (log_cost.evaluate(X[:, 0]) / 6.0).clamp(min=1e-4)
    with torch.no_grad():
        assert torch.allclose(weighted(X), unweighted(X) / c_norm)


@pytest.mark.parametrize(
    ("prior_mean", "mean_type"), [("power_law", PowerLawMean), ("zero", ZeroMean)]
)
def test_prior_mean_keeps_the_schedule(
    prior_mean: str, mean_type: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    gps: list[SingleTaskGP] = []
    make_gp = bo_module.make_single_obj_gp

    def spy(*args, **kwargs) -> SingleTaskGP:
        gps.append(make_gp(*args, **kwargs))
        return gps[-1]

    monkeypatch.setattr(bo_module, "make_single_obj_gp", spy)
    optimizer = _optimizer(
        "left_budget_contraction", prior_mean=prior_mean, acquisition_func_type="IT-JES"
    )
    budget_info = BudgetInfo(worker_cost_to_spend=8e6)
    trials: dict[str, Trial] = {}
    for _ in range(5):
        spent = sum(_cost(**t.config) for t in trials.values())
        cap = optimizer._schedule_cap(
            spent, 8e6, is_initial_design=len(trials) < optimizer.n_initial_design
        )
        sampled = optimizer(trials, budget_info)
        assert _cost(**sampled.config) <= cap
        trials[sampled.id] = _trial(sampled)

    assert gps
    assert all(type(gp.mean_module) is mean_type and gp.log_y for gp in gps)


@pytest.mark.parametrize(
    ("acquisition", "predictor"), [("EI", "power_law"), ("IT-SCAN-JES", "gp_posterior")]
)
def test_scan_predicts_with_its_gp(acquisition: str, predictor: str) -> None:
    from scalefm.optimizers import _make_info

    optimizer = bayesian_optimization(
        _space(), acquisition_func_type=acquisition, target_cost=10**6
    )
    assert optimizer.predictor == predictor
    # `scalefm.predict(predictor="auto")` reads it from the saved optimizer info
    info = _make_info("bayesian_optimization", {}, optimizer)
    assert info["derived"]["predictor"] == predictor
