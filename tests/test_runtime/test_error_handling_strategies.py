from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from pytest_cases import fixture, parametrize

from autoscan.exceptions import WorkerRaiseError
from autoscan.optimizers import OptimizerInfo
from autoscan.optimizers.algorithms import random_search
from autoscan.runtime import DefaultWorker
from autoscan.space.autoscan_spaces.parameters import Float, PipelineSpace
from autoscan.state import (
    AutoScAnState,
    DefaultReportValues,
    OnErrorPossibilities,
    OptimizationState,
    SeedSnapshot,
    Trial,
    WorkerSettings,
)


@fixture
def autoscan_state(tmp_path: Path) -> AutoScAnState:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    return AutoScAnState.create_or_load(
        path=tmp_path / "autoscan_state",
        optimizer_info=OptimizerInfo(name="blah", info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None,
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state=None,
        ),
        pipeline_space=TestSpace(),
    )


@parametrize(
    "on_error",
    [OnErrorPossibilities.RAISE_ANY_ERROR, OnErrorPossibilities.RAISE_WORKER_ERROR],
)
def test_worker_raises_when_error_in_self(
    autoscan_state: AutoScAnState,
    on_error: OnErrorPossibilities,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(TestSpace())
    settings = WorkerSettings(
        on_error=on_error,  # <- Highlight
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        worker_fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> float:
        raise ValueError("This is an error")

    worker = DefaultWorker.new(
        state=autoscan_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    with pytest.raises(WorkerRaiseError):
        worker.run()

    trials = autoscan_state.lock_and_read_trials()
    n_crashed = sum(
        trial.metadata.state == Trial.State.CRASHED is not None
        for trial in trials.values()
    )
    assert len(trials) == 1
    assert n_crashed == 1

    assert autoscan_state.lock_and_get_next_pending_trial() is None
    assert len(autoscan_state.lock_and_get_errors()) == 1


def test_worker_raises_when_error_in_other_worker(autoscan_state: AutoScAnState) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(TestSpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.RAISE_ANY_ERROR,  # <- Highlight
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        worker_fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def evaler(*args, **kwargs) -> float:
        raise ValueError("This is an error")

    worker1 = DefaultWorker.new(
        state=autoscan_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
    )
    worker2 = DefaultWorker.new(
        state=autoscan_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
    )

    # Worker1 should run 1 and error out
    with contextlib.suppress(WorkerRaiseError):
        worker1.run()

    # Worker2 should not run and immeditaly error out, however
    # it will have loaded in a serialized error
    with pytest.raises(WorkerRaiseError):
        worker2.run()

    trials = autoscan_state.lock_and_read_trials()
    n_crashed = sum(
        trial.metadata.state == Trial.State.CRASHED is not None
        for trial in trials.values()
    )
    assert len(trials) == 1
    assert n_crashed == 1

    assert autoscan_state.lock_and_get_next_pending_trial() is None
    assert len(autoscan_state.lock_and_get_errors()) == 1


@pytest.mark.parametrize(
    "on_error",
    [OnErrorPossibilities.IGNORE, OnErrorPossibilities.RAISE_WORKER_ERROR],
)
def test_worker_does_not_raise_when_error_in_other_worker(
    autoscan_state: AutoScAnState,
    on_error: OnErrorPossibilities,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(TestSpace())
    settings = WorkerSettings(
        on_error=on_error,  # <- Highlight
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        worker_fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    @dataclass
    class _Eval:
        do_raise: bool

        def __call__(self, *args, **kwargs) -> float:  # noqa: ARG002
            if self.do_raise:
                raise ValueError("This is an error")
            return 10

    evaler = _Eval(do_raise=True)

    worker1 = DefaultWorker.new(
        state=autoscan_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
    )
    worker2 = DefaultWorker.new(
        state=autoscan_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
    )

    # Worker1 should run 1 and error out
    evaler.do_raise = True
    with contextlib.suppress(WorkerRaiseError):
        worker1.run()

    trials = list(autoscan_state.lock_and_read_trials().values())
    assert (
        sum(1 for t in trials if t.metadata.evaluating_worker_id == worker1.worker_id)
        == 1
    )

    # Worker2 should run successfully
    evaler.do_raise = False
    worker2.run()
    trials = list(autoscan_state.lock_and_read_trials().values())
    assert (
        sum(1 for t in trials if t.metadata.evaluating_worker_id == worker2.worker_id)
        == 1
    )

    n_success = sum(
        trial.metadata.state == Trial.State.SUCCESS is not None for trial in trials
    )
    n_crashed = sum(
        trial.metadata.state == Trial.State.CRASHED is not None for trial in trials
    )
    assert n_success == 1
    assert n_crashed == 1
    assert len(trials) == 2

    assert autoscan_state.lock_and_get_next_pending_trial() is None
    assert len(autoscan_state.lock_and_get_errors()) == 1
