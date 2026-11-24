from __future__ import annotations

import math

import pytest
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from botorch.fit import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood

from scalefm.optimizers.models.gp import (
    LogCost,
    PowerLawMean,
    make_default_single_obj_gp,
    optimize_acq,
)
from scalefm.space import ConfigEncoder, HPOFloat
from scalefm.utils.power_law import fit_power_law_mean_params


def _cost(n_params: float, n_data: float) -> float:
    return 6.0 * n_params * n_data


def test_power_law_mean_matches_the_neps_checkpoints() -> None:
    encoder = ConfigEncoder.from_parameters(
        {
            "n_params": HPOFloat(1e6, 1e9, log=True),
            "n_data": HPOFloat(1e8, 1e11, log=True),
        }
    )
    x = torch.rand(5, 2, dtype=torch.float64)
    y = torch.rand(5, 1, dtype=torch.float64)
    gp = make_default_single_obj_gp(
        x,
        y,
        encoder,
        cost_estimator=_cost,
        min_cost=1e15,
        mean_slope=-0.1,
        mean_bias=2.0,
    )
    assert isinstance(gp.mean_module, PowerLawMean)
    # GP checkpoints written with NePS store the power law under these names
    keys = set(gp.state_dict())
    assert {
        "mean_module.fixed_slope",
        "mean_module.fixed_bias",
        "mean_module.reference_flops",
    } <= keys

    costs = torch.tensor([_cost(**c) for c in encoder.decode(x)], dtype=torch.float64)
    # The fitted law in the standardized outcome space of the GP
    loss = 10 ** (-0.1 * torch.log10(costs) + 2.0)
    expected = (loss - y.mean()) / y.std()
    assert torch.allclose(gp.mean_module(x), expected)


def test_power_law_mean_params_only_fit_the_pareto_front() -> None:
    def _loss(n_params: float, n_data: float) -> float:
        return 10.0 * _cost(n_params, n_data) ** -0.05

    front = [{"n_params": 10.0**k, "n_data": 1e9} for k in range(6, 10)]
    front_losses = [_loss(**c) for c in front]
    # Worse than a cheaper configuration on the front, so dominated
    dominated = [{"n_params": 10.0**k, "n_data": 2e9} for k in range(6, 10)]
    dominated_losses = [2 * front_losses[0]] * len(dominated)

    on_front = fit_power_law_mean_params(front, front_losses, _cost)
    with_dominated = fit_power_law_mean_params(
        front + dominated, front_losses + dominated_losses, _cost
    )
    assert with_dominated.fit is not None
    assert with_dominated.fit.num_points == len(front)
    assert with_dominated.slope == on_front.slope
    assert with_dominated.intercept == on_front.intercept
    assert abs(on_front.slope + 0.05) < 1e-9


def _cost_with_hps(n_params: float, n_data: float, **_: float) -> float:
    return _cost(n_params, n_data)


def _encoder() -> ConfigEncoder:
    return ConfigEncoder.from_parameters(
        {
            "lr": HPOFloat(1e-4, 1e-1, log=True),
            "n_params": HPOFloat(1e6, 1e9, log=True),
            "n_data": HPOFloat(1e8, 1e11, log=True),
        }
    )


def test_power_law_mean_with_log_y_is_the_standardized_log_law() -> None:
    encoder = _encoder()
    x = torch.rand(8, 3, dtype=torch.float64)
    y = torch.rand(8, 1, dtype=torch.float64) + 1.0
    gp = make_default_single_obj_gp(
        x,
        y,
        encoder,
        cost_estimator=_cost_with_hps,
        min_cost=1e15,
        mean_slope=-0.1,
        mean_bias=2.0,
        log_y=True,
    )
    assert gp.log_y
    costs = torch.tensor(
        [_cost_with_hps(**c) for c in encoder.decode(x)], dtype=torch.float64
    )
    log_loss = math.log(10) * (-0.1 * torch.log10(costs) + 2.0)
    expected = (log_loss - y.log().mean()) / y.log().std()
    assert torch.allclose(gp.mean_module(x), expected)
    # The GP models the outcomes as `Standardize` transforms their log
    log_y = y.log()
    expected_targets = ((log_y - log_y.mean()) / log_y.std()).squeeze(-1)
    assert torch.allclose(gp.train_targets, expected_targets)


def test_log_y_falls_back_for_non_positive_outcomes() -> None:
    encoder = _encoder()
    x = torch.rand(5, 3, dtype=torch.float64)
    y = torch.randn(5, 1, dtype=torch.float64)
    y[0] = -1.0
    gp = make_default_single_obj_gp(
        x, y, encoder, cost_estimator=_cost_with_hps, log_y=True
    )
    assert not gp.log_y


def test_log_cost_has_finite_difference_gradients() -> None:
    encoder = _encoder()
    log_cost = LogCost(encoder, _cost_with_hps)
    x = torch.full((1, 3), 0.5, dtype=torch.float64, requires_grad=True)
    f = log_cost(x)
    assert f.shape == (1,)
    f.sum().backward()
    grad = x.grad[0]
    # log10(cost) is linear in the log encoded n_params and n_data, with slope the
    # number of decades they span, and does not depend on the lr
    assert grad[encoder.index_of["lr"]] == 0
    assert grad[encoder.index_of["n_params"]] == pytest.approx(3.0, rel=1e-3)
    assert grad[encoder.index_of["n_data"]] == pytest.approx(3.0, rel=1e-3)


def test_power_law_mean_gp_fits_and_acquires() -> None:
    encoder = _encoder()
    x = torch.rand(12, 3, dtype=torch.float64)
    costs = torch.tensor(
        [_cost_with_hps(**c) for c in encoder.decode(x)], dtype=torch.float64
    )
    y = (10 ** (-0.05 * torch.log10(costs) + 1.0)).unsqueeze(-1)
    gp = make_default_single_obj_gp(
        x,
        y,
        encoder,
        cost_estimator=_cost_with_hps,
        mean_slope=-0.05,
        mean_bias=1.0,
        log_y=True,
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp))
    acq = qLogNoisyExpectedImprovement(
        model=gp,
        X_baseline=x,
        objective=LinearMCObjective(weights=torch.tensor([-1.0], dtype=torch.float64)),
    )
    candidates, _ = optimize_acq(acq, encoder, num_restarts=2, n_intial_start_points=32)
    assert candidates.shape == (1, 3)
    assert ((candidates >= 0) & (candidates <= 1)).all()
