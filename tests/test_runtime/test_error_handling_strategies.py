from __future__ import annotations

import pytest
import os
from dataclasses import dataclass
from pandas.core.common import contextlib
import signal
from pathlib import Path
from pytest_cases import fixture, parametrize
import multiprocessing
import psutil
import time

from neps.optimizers.random_search.optimizer import RandomSearch
from neps.runtime import DefaultWorker, SIGNALS_TO_HANDLE_IF_AVAILABLE
from neps.search_spaces.search_space import SearchSpace
from neps.state.err_dump import SerializedError
from neps.state.filebased import create_or_load_filebased_neps_state
from neps.state.neps_state import NePSState
from neps.state.optimizer import OptimizationState, OptimizerInfo
from neps.state.settings import DefaultReportValues, OnErrorPossibilities, WorkerSettings
from neps.search_spaces import FloatParameter
from neps.state.trial import Trial


@fixture
def neps_state(tmp_path: Path) -> NePSState[Path]:
    return create_or_load_filebased_neps_state(
        directory=tmp_path / "neps_state",
        optimizer_info=OptimizerInfo(info={"nothing": "here"}),
        optimizer_state=OptimizationState(budget=None, shared_state={}),
    )


@parametrize(
    "on_error",
    [OnErrorPossibilities.RAISE_ANY_ERROR, OnErrorPossibilities.RAISE_WORKER_ERROR],
)
def test_worker_raises_when_error_in_self(
    neps_state: NePSState,
    on_error: OnErrorPossibilities,
) -> None:
    optimizer = RandomSearch(pipeline_space=SearchSpace(a=FloatParameter(0, 1)))
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
    )

    def eval_function(*args, **kwargs) -> float:
        raise ValueError("This is an error")

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    with pytest.raises(ValueError, match="This is an error"):
        worker.run()

    trials = neps_state.get_all_trials()
    n_crashed = sum(
        trial.state == Trial.State.CRASHED is not None for trial in trials.values()
    )
    assert len(trials) == 1
    assert n_crashed == 1

    assert neps_state.get_next_pending_trial() is None
    assert len(neps_state.get_errors()) == 1


def test_worker_raises_when_error_in_other_worker(neps_state: NePSState) -> None:
    optimizer = RandomSearch(pipeline_space=SearchSpace(a=FloatParameter(0, 1)))
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
    )

    def evaler(*args, **kwargs) -> float:
        raise ValueError("This is an error")

    worker1 = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
        _pre_sample_hooks=None,
    )
    worker2 = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
        _pre_sample_hooks=None,
    )

    # Worker1 should run 1 and error out
    with contextlib.suppress(ValueError):
        worker1.run()

    # Worker2 should not run and immeditaly error out, however
    # it will have loaded in a serialized error
    with pytest.raises(SerializedError):
        worker2.run()

    trials = neps_state.get_all_trials()
    n_crashed = sum(
        trial.state == Trial.State.CRASHED is not None for trial in trials.values()
    )
    assert len(trials) == 1
    assert n_crashed == 1

    assert neps_state.get_next_pending_trial() is None
    assert len(neps_state.get_errors()) == 1


def test_worker_does_not_raise_when_error_in_other_worker(neps_state: NePSState) -> None:
    optimizer = RandomSearch(pipeline_space=SearchSpace(a=FloatParameter(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.RAISE_WORKER_ERROR,  # <- Highlight
        default_report_values=DefaultReportValues(),
        max_evaluations_total=None,
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=1,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
    )

    @dataclass
    class _Eval:
        do_raise: bool

        def __call__(self, *args, **kwargs) -> float:
            if self.do_raise:
                raise ValueError("This is an error")
            return 10

    evaler = _Eval(do_raise=True)

    worker1 = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
        _pre_sample_hooks=None,
    )
    worker2 = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=evaler,
        settings=settings,
        _pre_sample_hooks=None,
    )

    # Worker1 should run 1 and error out
    evaler.do_raise = True
    with contextlib.suppress(ValueError):
        worker1.run()
    assert worker1.worker_cumulative_eval_count == 1

    # Worker2 should run successfully
    evaler.do_raise = False
    worker2.run()
    assert worker2.worker_cumulative_eval_count == 1

    trials = neps_state.get_all_trials()
    n_success = sum(
        trial.state == Trial.State.SUCCESS is not None for trial in trials.values()
    )
    n_crashed = sum(
        trial.state == Trial.State.CRASHED is not None for trial in trials.values()
    )
    assert n_success == 1
    assert n_crashed == 1
    assert len(trials) == 2

    assert neps_state.get_next_pending_trial() is None
    assert len(neps_state.get_errors()) == 1


def sleep_function(*args, **kwargs) -> float:
    time.sleep(10)
    return 10


SIGNALS: list[signal.Signals] = []
for name in SIGNALS_TO_HANDLE_IF_AVAILABLE:
    if hasattr(signal.Signals, name):
        sig: signal.Signals = getattr(signal.Signals, name)
        SIGNALS.append(sig)


@pytest.mark.ci_examples
@pytest.mark.parametrize("signum", SIGNALS)
def test_worker_reset_evaluating_to_pending_on_ctrl_c(
    signum: signal.Signals,
    neps_state: NePSState,
) -> None:
    optimizer = RandomSearch(pipeline_space=SearchSpace(a=FloatParameter(0, 1)))
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
    )

    worker1 = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=sleep_function,
        settings=settings,
        _pre_sample_hooks=None,
    )

    # Use multiprocessing.Process
    p = multiprocessing.Process(
        target=worker1.run, args=(neps_state, optimizer, settings)
    )
    p.start()

    time.sleep(5)
    assert p.pid is not None
    assert p.is_alive()

    # Should be evaluating at this stage
    trials = neps_state.get_all_trials()
    assert len(trials) == 1
    assert next(iter(trials.values())).state == Trial.State.EVALUATING

    # Kill the process while it's evaluating using signals
    process = psutil.Process(p.pid)
    process.send_signal(signum)
    p.join(timeout=10)  # Wait for the process to terminate

    if p.is_alive():
        p.terminate()  # Force terminate if it's still alive
        p.join()
        pytest.fail("Worker did not terminate after receiving signal!")
    else:
        trials2 = neps_state.get_all_trials()
        assert len(trials2) == 1
        assert next(iter(trials2.values())).state == Trial.State.PENDING
