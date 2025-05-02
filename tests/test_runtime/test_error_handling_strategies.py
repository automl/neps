from __future__ import annotations

import contextlib
import multiprocessing
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import psutil
import pytest
from pytest_cases import fixture, parametrize

from neps.exceptions import WorkerRaiseError
from neps.optimizers import OptimizerInfo
from neps.optimizers.algorithms import random_search
from neps.runtime import SIGNALS_TO_HANDLE_IF_AVAILABLE, DefaultWorker
from neps.space import Float, SearchSpace
from neps.state import (
    DefaultReportValues,
    NePSState,
    OnErrorPossibilities,
    OptimizationState,
    SeedSnapshot,
    Trial,
    WorkerSettings,
)


@fixture
def neps_state(tmp_path: Path) -> NePSState:
    return NePSState.create_or_load(
        path=tmp_path / "neps_state",
        optimizer_info=OptimizerInfo(name="blah", info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None,
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state=None,
        ),
    )


@parametrize(
    "on_error",
    [OnErrorPossibilities.RAISE_ANY_ERROR, OnErrorPossibilities.RAISE_WORKER_ERROR],
)
def test_worker_raises_when_error_in_self(
    neps_state: NePSState,
    on_error: OnErrorPossibilities,
) -> None:
    optimizer = random_search(SearchSpace({"a": Float(0, 1)}))
    settings = WorkerSettings(
        on_error=on_error,  # <- Highlight
        default_report_values=DefaultReportValues(),
        max_evaluations_total=None,
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=1,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> float:
        raise ValueError("This is an error")

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    with pytest.raises(WorkerRaiseError):
        worker.run()

    trials = neps_state.lock_and_read_trials()
    n_crashed = sum(
        trial.metadata.state == Trial.State.CRASHED is not None
        for trial in trials.values()
    )
    assert len(trials) == 1
    assert n_crashed == 1

    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 1


def test_worker_raises_when_error_in_other_worker(neps_state: NePSState) -> None:
    optimizer = random_search(SearchSpace({"a": Float(0, 1)}))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.RAISE_ANY_ERROR,  # <- Highlight
        default_report_values=DefaultReportValues(),
        max_evaluations_total=None,
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=1,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
    )

    def evaler(*args, **kwargs) -> float:
        raise ValueError("This is an error")

    worker1 = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
    )
    worker2 = DefaultWorker.new(
        state=neps_state,
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

    trials = neps_state.lock_and_read_trials()
    n_crashed = sum(
        trial.metadata.state == Trial.State.CRASHED is not None
        for trial in trials.values()
    )
    assert len(trials) == 1
    assert n_crashed == 1

    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 1


@pytest.mark.parametrize(
    "on_error",
    [OnErrorPossibilities.IGNORE, OnErrorPossibilities.RAISE_WORKER_ERROR],
)
def test_worker_does_not_raise_when_error_in_other_worker(
    neps_state: NePSState,
    on_error: OnErrorPossibilities,
) -> None:
    optimizer = random_search(SearchSpace({"a": Float(0, 1)}))
    settings = WorkerSettings(
        on_error=on_error,  # <- Highlight
        default_report_values=DefaultReportValues(),
        max_evaluations_total=None,
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=1,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
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
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
    )
    worker2 = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
    )

    # Worker1 should run 1 and error out
    evaler.do_raise = True
    with contextlib.suppress(WorkerRaiseError):
        worker1.run()
    assert worker1.worker_cumulative_eval_count == 1

    # Worker2 should run successfully
    evaler.do_raise = False
    worker2.run()
    assert worker2.worker_cumulative_eval_count == 1

    trials = neps_state.lock_and_read_trials()
    n_success = sum(
        trial.metadata.state == Trial.State.SUCCESS is not None
        for trial in trials.values()
    )
    n_crashed = sum(
        trial.metadata.state == Trial.State.CRASHED is not None
        for trial in trials.values()
    )
    assert n_success == 1
    assert n_crashed == 1
    assert len(trials) == 2

    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 1


def sleep_function(*args, **kwargs) -> float:
    time.sleep(20)
    return 10


SIGNALS: list[signal.Signals] = []
for name in SIGNALS_TO_HANDLE_IF_AVAILABLE:
    if hasattr(signal.Signals, name):
        sig: signal.Signals = getattr(signal.Signals, name)
        SIGNALS.append(sig)


# @pytest.mark.ci_examples
@pytest.mark.parametrize("signum", SIGNALS)
def test_worker_reset_evaluating_to_pending_on_ctrl_c(
    signum: signal.Signals,
    neps_state: NePSState,
) -> None:
    optimizer = random_search(SearchSpace({"a": Float(0, 1)}))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,  # <- Highlight
        default_report_values=DefaultReportValues(),
        max_evaluations_total=None,
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=1,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
    )

    worker1 = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=sleep_function,
        settings=settings,
    )

    # Use multiprocessing.Process
    p = multiprocessing.Process(target=worker1.run)
    p.start()

    time.sleep(5)
    assert p.pid is not None
    assert p.is_alive()

    # Should be evaluating at this stage
    trials = neps_state.lock_and_read_trials()
    assert len(trials) == 1
    assert next(iter(trials.values())).metadata.state == Trial.State.EVALUATING

    # Kill the process while it's evaluating using signals
    process = psutil.Process(p.pid)

    # If sending the signal fails, skip the test,
    # as most likely the signal is not supported on this platform
    try:
        process.send_signal(signum)
    except ValueError as e:
        pytest.skip(f"Signal error: {e}")
    else:
        # If the signal is sent successfully, we can proceed with the test
        pass

    # If the system is windows and the signal is SIGTERM, skip the test
    if (
        signum == signal.SIGTERM
        and multiprocessing.get_start_method() == "spawn"
        and multiprocessing.current_process().name == "MainProcess"
    ):
        pytest.skip("SIGTERM is not supported on Windows with spawn start method")

    p.join(timeout=5)  # Wait for the process to terminate

    if p.is_alive():
        p.terminate()  # Force terminate if it's still alive
        p.join()
        pytest.fail("Worker did not terminate after receiving signal!")
    else:
        trials2 = neps_state.lock_and_read_trials()
        assert len(trials2) == 1
        assert next(iter(trials2.values())).metadata.state == Trial.State.PENDING
