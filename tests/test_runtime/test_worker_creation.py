from __future__ import annotations

from pathlib import Path

import pytest

from autoscan import Float, PipelineSpace
from autoscan.optimizers import OptimizerInfo
from autoscan.optimizers.algorithms import random_search
from autoscan.runtime import (
    DefaultReportValues,
    DefaultWorker,
    OnErrorPossibilities,
    WorkerSettings,
)
from autoscan.state import AutoScAnState, OptimizationState, SeedSnapshot


@pytest.fixture
def autoscan_state(tmp_path: Path) -> AutoScAnState:
    return AutoScAnState.create_or_load(
        path=tmp_path / "autoscan_state",
        optimizer_info=OptimizerInfo(name="blah", info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None, seed_snapshot=SeedSnapshot.new_capture(), shared_state={}
        ),
        pipeline_space=ASpace(),
    )


class ASpace(PipelineSpace):
    a = Float(0, 1)


def test_create_worker_manual_id(autoscan_state: AutoScAnState) -> None:
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        worker_fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_fn(config: dict) -> float:
        return 1.0

    test_worker_id = "my_worker_123"

    optimizer = random_search(ASpace())

    worker = DefaultWorker.new(
        state=autoscan_state,
        settings=settings,
        optimizer=optimizer,
        evaluation_fn=eval_fn,
        worker_id=test_worker_id,
    )

    assert worker.worker_id == test_worker_id
    assert autoscan_state.lock_and_get_optimizer_state().worker_ids == [test_worker_id]


def test_create_worker_auto_id(autoscan_state: AutoScAnState) -> None:
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        worker_fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_fn(config: dict) -> float:
        return 1.0

    optimizer = random_search(ASpace())

    worker = DefaultWorker.new(
        state=autoscan_state,
        settings=settings,
        optimizer=optimizer,
        evaluation_fn=eval_fn,
    )

    assert worker.worker_id == "worker_0"
    assert autoscan_state.lock_and_get_optimizer_state().worker_ids == [worker.worker_id]
