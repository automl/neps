"""Tests for `extrapolaw.acquire(..., live_plots=True)`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import extrapolaw
from extrapolaw.space.extrapolaw_spaces.parameters import Float, PipelineSpace


class _Space(PipelineSpace):
    x = Float(0.0, 1.0)


def _single_objective(x: float) -> dict:
    return {"objective_to_minimize": x, "cost": 1.0}


def _two_objectives(x: float) -> dict:
    return {"objective_to_minimize": [x, 1 - x], "cost": 1.0}


def _run(root: Path, evaluate_pipeline: object, *, live_plots: bool) -> None:
    extrapolaw.acquire(
        evaluate_pipeline=evaluate_pipeline,  # type: ignore[arg-type]
        pipeline_space=_Space(),
        root_directory=root,
        optimizer="random_search",
        worker_evaluations_to_spend=3,
        live_plots=live_plots,
    )


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

    assert not list((root / "summary").rglob("*.png"))
