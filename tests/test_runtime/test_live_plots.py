"""Tests for `extrapolaw.acquire(..., live_plots=True)`."""

from __future__ import annotations

from pathlib import Path

import extrapolaw
from extrapolaw.space.extrapolaw_spaces.parameters import Float, PipelineSpace


class _Space(PipelineSpace):
    x = Float(0.0, 1.0)


def _single_objective(x: float) -> dict:
    return {"objective_to_minimize": x, "cost": 1.0}


def _run(root: Path, evaluate_pipeline: object, *, live_plots: bool) -> None:
    extrapolaw.acquire(
        evaluate_pipeline=evaluate_pipeline,  # type: ignore[arg-type]
        pipeline_space=_Space(),
        root_directory=root,
        optimizer="random_search",
        worker_evaluations_to_spend=3,
        live_plots=live_plots,
    )


def test_no_plots_by_default(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _run(root, _single_objective, live_plots=False)

    assert not list((root / "summary").rglob("*.png"))
