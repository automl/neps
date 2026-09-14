"""Tests for `neps.run(..., live_plots=True)`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import neps
from neps.space.neps_spaces.parameters import Float, PipelineSpace


class _Space(PipelineSpace):
    x = Float(0.0, 1.0)


def _single_objective(x: float) -> dict:
    return {"objective_to_minimize": x, "cost": 1.0}


def _two_objectives(x: float) -> dict:
    return {"objective_to_minimize": [x, 1 - x]}


def _run(root: Path, evaluate_pipeline: object, *, live_plots: bool) -> None:
    neps.run(
        evaluate_pipeline=evaluate_pipeline,  # type: ignore[arg-type]
        pipeline_space=_Space(),
        root_directory=root,
        optimizer="random_search",
        evaluations_to_spend=3,
        live_plots=live_plots,
    )


def test_incumbent_trajectory_is_written(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _run(root, _single_objective, live_plots=True)

    assert (root / "summary" / "incumbent_trajectory.png").exists()
    table = pd.read_csv(root / "summary" / "incumbent_trajectory.csv")
    assert len(table) == 3
    assert table["cumulative_cost"].tolist() == [1.0, 2.0, 3.0]
    assert table["incumbent_objective_to_minimize"].is_monotonic_decreasing
    assert table["is_incumbent"].iloc[0]


def test_pareto_front_is_written(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _run(root, _two_objectives, live_plots=True)

    assert (root / "summary" / "pareto_front.png").exists()
    table = pd.read_csv(root / "summary" / "pareto_front.csv")
    assert len(table) == 3
    # All points trade off `x` against `1 - x`, so none dominates another
    assert table["on_pareto_front"].all()


def test_no_plots_by_default(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _run(root, _single_objective, live_plots=False)

    assert not list((root / "summary").glob("*.png"))
    assert not (root / "summary" / "incumbent_trajectory.csv").exists()
