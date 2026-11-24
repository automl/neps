from __future__ import annotations

import json
from pathlib import Path

import pytest

import scalefm
from scalefm.optimizers import load_optimizer
from scalefm.optimizers.algorithms import grid_search, random_search
from scalefm.space import HPOCategorical, HPOFloat, HPOInteger, SearchSpace
from scalefm.state import ScaleFMState
from scalefm.utils.cost_estimator import CostEstimator


class Space(scalefm.PipelineSpace):
    width = scalefm.Categorical(choices=(64, 128, 256))
    lr = scalefm.Float(1e-4, 1e-1, log=True)
    dataset = "toy"


def _costs(root: Path) -> dict[str, float]:
    trials = ScaleFMState.create_or_load(root, load_only=True).lock_and_read_trials()
    return {t.id: t.report.cost for t in trials.values() if t.report is not None}


def test_estimated_cost_is_reported_and_spent(tmp_path: Path) -> None:
    def evaluate_pipeline(width: int, lr: float, dataset: str) -> dict:
        return {"objective_to_minimize": lr, "cost": width}

    scalefm.acquire(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=Space(),
        root_directory=tmp_path / "run",
        optimizer="random_search",
        cost_estimator=float,
        total_cost_to_spend=600,
    )
    costs = _costs(tmp_path / "run")
    assert set(costs.values()) <= {64.0, 128.0, 256.0}
    assert 600 <= sum(costs.values()) < 600 + 256


def test_reported_cost_is_kept(tmp_path: Path) -> None:
    table = tmp_path / "costs.json"
    table.write_text(
        json.dumps([{"config": {"width": w}, "cost": w * 10} for w in (64, 128, 256)])
    )

    def evaluate_pipeline(
        width: int, lr: float, dataset: str, pipeline_directory: Path
    ) -> dict:
        assert pipeline_directory.exists()
        return {"objective_to_minimize": lr, "cost": 1.0 if width == 64 else None}

    scalefm.acquire(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=Space(),
        root_directory=tmp_path / "run",
        optimizer=("grid_search", {"size_per_numerical_dimension": 1}),
        cost_estimator=table,
        worker_evaluations_to_spend=3,
    )
    assert sorted(_costs(tmp_path / "run").values()) == [1.0, 1280.0, 2560.0]


def test_invalid_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a `cost_estimator`"):
        scalefm.acquire(
            evaluate_pipeline=lambda **_: 1.0,
            pipeline_space=Space(),
            root_directory=tmp_path / "run",
            max_cost_per_config=10,
        )

    table = tmp_path / "costs.json"
    table.write_text(json.dumps([{"config": {"depth": 1}, "cost": 1.0}]))
    with pytest.raises(ValueError, match="not in the pipeline space"):
        scalefm.acquire(
            evaluate_pipeline=lambda **_: 1.0,
            pipeline_space=Space(),
            root_directory=tmp_path / "run",
            cost_estimator=table,
        )


def test_max_cost_per_config_constrains_random_and_grid_search() -> None:
    space = SearchSpace(
        {
            "width": HPOInteger(1, 100),
            "lr": HPOFloat(0.0, 1.0),
            "opt": HPOCategorical(["adam", "sgd"]),
        }
    )

    def cost_estimator(**config: dict) -> float:
        return config["width"]

    estimator = CostEstimator.from_callable(cost_estimator)

    random, _ = load_optimizer(
        random_search, space, cost_estimator=estimator, max_cost_per_config=20
    )
    assert all(s.config["width"] <= 20 for s in random({}, None, n=20))

    grid, _ = load_optimizer(
        grid_search, space, cost_estimator=estimator, max_cost_per_config=20
    )
    assert grid({}, None).config["width"] <= 20
    assert any(c["width"] > 20 for c in grid.configs_list)


def test_bayesian_optimization_without_cost_estimator(tmp_path: Path) -> None:
    class FloatSpace(scalefm.PipelineSpace):
        x = scalefm.Float(0.0, 1.0)
        y = scalefm.Float(0.0, 1.0)

    scalefm.acquire(
        evaluate_pipeline=lambda x, y: {
            "objective_to_minimize": (x - 0.3) ** 2 + y,
            "cost": 1.0,
        },
        pipeline_space=FloatSpace(),
        root_directory=tmp_path / "run",
        optimizer=("bayesian_optimization", {"initial_design_size": 3}),
        worker_evaluations_to_spend=5,
    )
    assert len(_costs(tmp_path / "run")) == 5
