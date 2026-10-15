from __future__ import annotations

import time

import pytest
import torch

from autoscan.optimizers.algorithms import bayesian_optimization
from autoscan.optimizers.models.gp import (
    INFEASIBLE_CONSTRAINT_VALUE,
    encode_constraints_func,
    make_ic_generator,
)
from autoscan.optimizers.optimizer import SampledConfig
from autoscan.optimizers.utils.initial_design import make_initial_design
from autoscan.space import (
    HPOCategorical,
    HPOConstant,
    HPOFloat,
    HPOInteger,
    SearchSpace,
)
from autoscan.space.encoding import ConfigEncoder
from autoscan.state import Trial


def _trial(sampled: SampledConfig, objective: float) -> Trial:
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
        cost=None,
        learning_curve=None,
        err=None,
        tb=None,
        extra=None,
        evaluation_duration=0.0,
    )
    return trial


def _objective(config: dict) -> float:
    return (config["x"] - 0.8) ** 2 + (config["width"] - 60) ** 2 / 1e4


def _run(optimizer, steps: int, n: int | None = None) -> list[dict]:
    trials: dict[str, Trial] = {}
    configs: list[dict] = []
    for _ in range(steps):
        sampled = optimizer(trials, None, n=n)
        sampled = [sampled] if isinstance(sampled, SampledConfig) else sampled
        for s in sampled:
            trials[s.id] = _trial(s, _objective(s.config))
            configs.append(s.config)
    return configs


def _numeric_space() -> SearchSpace:
    return SearchSpace(
        {
            "x": HPOFloat(0.0, 1.0),
            "width": HPOInteger(1, 100),
            "dataset": HPOConstant("toy"),
        }
    )


def _constraint(config: dict) -> float:
    # The unconstrained optimum (x=0.8, width=60) is infeasible
    return 1.0 - config["x"] - config["width"] / 100


@pytest.mark.parametrize("n", [None, 3])
def test_numeric_space_only_proposes_feasible_configs(n: int | None) -> None:
    torch.manual_seed(0)
    optimizer = bayesian_optimization(
        _numeric_space(),
        initial_design_size=4,
        num_restarts=4,
        n_initial_start_points=64,
        constraints_func=_constraint,
    )
    configs = _run(optimizer, steps=3, n=n)
    assert all(_constraint(c) >= 0 for c in configs)


def test_mixed_space_with_constraints_is_supported() -> None:
    torch.manual_seed(0)
    space = SearchSpace(
        {
            "x": HPOFloat(0.0, 1.0),
            "width": HPOInteger(1, 100),
            "opt": HPOCategorical(["adam", "sgd", "lion"]),
        }
    )

    def constraint(config: dict) -> float:
        penalty = 0.3 if config["opt"] == "sgd" else 0.0
        return 1.0 - config["x"] - config["width"] / 100 - penalty

    optimizer = bayesian_optimization(
        space,
        initial_design_size=4,
        num_restarts=4,
        n_initial_start_points=64,
        constraints_func=constraint,
    )
    configs = _run(optimizer, steps=3)
    assert all(constraint(c) >= 0 for c in configs)


def test_budget_constraint_combines_cost_and_user_constraint_with_constants() -> None:
    optimizer = bayesian_optimization(
        _numeric_space(),
        # Needs the constant, which the GP never sees
        constraints_func=lambda config: 0.0 if config["dataset"] == "toy" else -1.0,
    )
    optimizer.set_cost_estimator(
        lambda **config: config["width"] * (2 if config["dataset"] == "toy" else 1),
        max_cost_per_config=100,
    )
    constraint = optimizer.get_constraint_func()
    assert constraint is not None
    assert constraint({"x": 0.5, "width": 50}) == 0.0
    assert constraint({"x": 0.5, "width": 51}) < 0


def test_initial_design_tolerates_infeasible_draws() -> None:
    space = _numeric_space()
    configs = make_initial_design(
        parameters=space.searchables,
        encoder=ConfigEncoder.from_parameters(space.searchables),
        sampler="uniform",
        sample_size=5,
        sample_prior_first=False,
        constraints_func=_constraint,
        oversampling_factor=40,
    )
    assert len(configs) == 5
    assert all(_constraint(c) >= 0 for c in configs)


def test_initial_design_keeps_prior_config_without_constraints() -> None:
    space = _numeric_space()
    configs = make_initial_design(
        parameters=space.searchables,
        encoder=ConfigEncoder.from_parameters(space.searchables),
        sampler="uniform",
        sample_size=3,
        sample_prior_first=True,
    )
    assert len(configs) == 4


def test_ic_generator_respects_constraints_and_fixed_features() -> None:
    space = _numeric_space()
    encoder = ConfigEncoder.from_parameters(space.searchables)
    constraint = encode_constraints_func(_constraint, encoder=encoder)
    bounds = torch.tensor(
        [[d.lower for d in encoder.domains], [d.upper for d in encoder.domains]],
        dtype=torch.float64,
    )
    x_idx = encoder.index_of["x"]
    ic_generator = make_ic_generator([(constraint, True)])

    ics = ic_generator(
        acq_function=lambda X: -X.sum(dim=(-1, -2)),
        bounds=bounds,
        q=2,
        num_restarts=5,
        raw_samples=64,
        fixed_features={x_idx: 0.25},
    )

    assert ics.shape == (5, 2, encoder.ndim)
    assert torch.all(ics[..., x_idx] == 0.25)
    assert torch.all(constraint(ics) >= 0)


def test_constraint_gradient_is_nonzero_for_integer_dims() -> None:
    space = _numeric_space()
    encoder = ConfigEncoder.from_parameters(space.searchables)
    constraint = encode_constraints_func(_constraint, encoder=encoder)

    x = encoder.encode([{"x": 0.2, "width": 30}]).requires_grad_(True)  # noqa: FBT003
    constraint(x).sum().backward()

    assert x.grad is not None
    assert x.grad[0, encoder.index_of["width"]] < 0
    assert x.grad[0, encoder.index_of["x"]] < 0


def test_failing_constraint_is_treated_as_infeasible() -> None:
    space = _numeric_space()
    encoder = ConfigEncoder.from_parameters(space.searchables)

    def broken(config: dict) -> float:
        raise KeyError("missing")

    constraint = encode_constraints_func(broken, encoder=encoder)
    x = encoder.encode([{"x": 0.2, "width": 30}])
    assert constraint(x).item() == INFEASIBLE_CONSTRAINT_VALUE
