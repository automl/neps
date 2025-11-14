from __future__ import annotations

from pathlib import Path

import pytest

from neps.optimizers import OptimizerInfo
from neps.optimizers.algorithms import random_search
from neps.runtime import (
    DefaultReportValues,
    DefaultWorker,
    OnErrorPossibilities,
    WorkerSettings,
)
from neps.space import Float, SearchSpace
from neps.state import NePSState, OptimizationState, RNGStateManager


@pytest.fixture
def neps_state(tmp_path: Path) -> NePSState:
    return NePSState.create_or_load(
        path=tmp_path / "neps_state",
        optimizer_info=OptimizerInfo(name="blah", info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None, rng_state_manager=RNGStateManager.new_capture(), shared_state={}
        ),
    )


def test_create_worker_manual_id(neps_state: NePSState) -> None:
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=True,
        cost_to_spend=None,
        max_evaluations_for_worker=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
        fidelities_to_spend=None,
    )

    def eval_fn(config: dict) -> float:
        return 1.0

    test_worker_id = "my_worker_123"
    optimizer = random_search(
        SearchSpace({"a": Float(0, 1)}), rng_manager=RNGStateManager.new_capture()
    )

    worker = DefaultWorker.new(
        state=neps_state,
        settings=settings,
        optimizer=optimizer,
        evaluation_fn=eval_fn,
        worker_id=test_worker_id,
    )

    assert worker.worker_id == test_worker_id
    assert neps_state.lock_and_get_optimizer_state().worker_ids == [test_worker_id]


def test_create_worker_auto_id(neps_state: NePSState) -> None:
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=True,
        cost_to_spend=None,
        max_evaluations_for_worker=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
        fidelities_to_spend=None,
    )

    def eval_fn(config: dict) -> float:
        return 1.0

    optimizer = random_search(
        SearchSpace({"a": Float(0, 1)}), rng_manager=RNGStateManager.new_capture()
    )

    worker = DefaultWorker.new(
        state=neps_state,
        settings=settings,
        optimizer=optimizer,
        evaluation_fn=eval_fn,
    )

    assert worker.worker_id == "worker_0"
    assert neps_state.lock_and_get_optimizer_state().worker_ids == [worker.worker_id]
