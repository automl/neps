from __future__ import annotations

from pathlib import Path

import pytest

from scalefm import Float, PipelineSpace
from scalefm.optimizers import OptimizerInfo
from scalefm.optimizers.algorithms import random_search
from scalefm.runtime import (
    DefaultReportValues,
    DefaultWorker,
    OnErrorPossibilities,
    WorkerSettings,
)
from scalefm.state import OptimizationState, ScaleFMState, SeedSnapshot


@pytest.fixture
def scalefm_state(tmp_path: Path) -> ScaleFMState:
    return ScaleFMState.create_or_load(
        path=tmp_path / "scalefm_state",
        optimizer_info=OptimizerInfo(name="blah", info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None, seed_snapshot=SeedSnapshot.new_capture(), shared_state={}
        ),
        pipeline_space=ASpace(),
    )


class ASpace(PipelineSpace):
    a = Float(0, 1)


def test_create_worker_manual_id(scalefm_state: ScaleFMState) -> None:
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_fn(config: dict) -> float:
        return 1.0

    test_worker_id = "my_worker_123"

    optimizer = random_search(ASpace())

    worker = DefaultWorker.new(
        state=scalefm_state,
        settings=settings,
        optimizer=optimizer,
        evaluation_fn=eval_fn,
        worker_id=test_worker_id,
    )

    assert worker.worker_id == test_worker_id
    assert scalefm_state.lock_and_get_optimizer_state().worker_ids == [test_worker_id]


def test_create_worker_auto_id(scalefm_state: ScaleFMState) -> None:
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_fn(config: dict) -> float:
        return 1.0

    optimizer = random_search(ASpace())

    worker = DefaultWorker.new(
        state=scalefm_state,
        settings=settings,
        optimizer=optimizer,
        evaluation_fn=eval_fn,
    )

    assert worker.worker_id == "worker_0"
    assert scalefm_state.lock_and_get_optimizer_state().worker_ids == [worker.worker_id]
