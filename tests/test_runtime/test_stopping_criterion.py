from __future__ import annotations

import time
from pathlib import Path

from pytest_cases import fixture

from neps.optimizers import random_search
from neps.runtime import DefaultWorker
from neps.search_spaces import Float, SearchSpace
from neps.state import (
    NePSState,
    OptimizationState,
    OptimizerInfo,
    SeedSnapshot,
    DefaultReportValues,
    OnErrorPossibilities,
    WorkerSettings,
    Trial,
)


@fixture
def neps_state(tmp_path: Path) -> NePSState:
    return NePSState.create_or_load(
        path=tmp_path / "neps_state",
        optimizer_info=OptimizerInfo(info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None,
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state=None,
        ),
    )


def test_max_evaluations_total_stopping_criterion(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        max_evaluations_total=3,  # <- Highlight
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> float:
        return 1.0

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    worker.run()

    assert worker.worker_cumulative_eval_count == 3
    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 0

    trials = neps_state.lock_and_read_trials()
    for _, trial in trials.items():
        assert trial.metadata.state == Trial.State.SUCCESS
        assert trial.report is not None
        assert trial.report.objective_to_minimize == 1.0

    # New worker has the same total number of evaluations so it should not run anything.
    new_worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    new_worker.run()
    assert new_worker.worker_cumulative_eval_count == 0
    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 0


def test_worker_evaluations_total_stopping_criterion(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        max_evaluations_total=None,
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=2,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> float:
        return 1.0

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    worker.run()

    assert worker.worker_cumulative_eval_count == 2
    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 0

    trials = neps_state.lock_and_read_trials()
    assert len(trials) == 2
    for _, trial in trials.items():
        assert trial.metadata.state == Trial.State.SUCCESS
        assert trial.report is not None
        assert trial.report.objective_to_minimize == 1.0

    # New worker should run 2 more evaluations
    new_worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    new_worker.run()

    assert worker.worker_cumulative_eval_count == 2
    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 0

    trials = neps_state.lock_and_read_trials()
    assert len(trials) == 4  # Now we should have 4 of them
    for _, trial in trials.items():
        assert trial.metadata.state == Trial.State.SUCCESS
        assert trial.report is not None
        assert trial.report.objective_to_minimize == 1.0


def test_include_in_progress_evaluations_towards_maximum_with_work_eval_count(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        max_evaluations_total=2,  # <- Highlight, only 2 maximum evaluations allowed
        include_in_progress_evaluations_towards_maximum=True,  # <- include the inprogress trial
        max_cost_total=None,
        max_evaluations_for_worker=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
    )

    # We put in one trial as being inprogress
    pending_trial = neps_state.lock_and_sample_trial(optimizer, worker_id="dummy")
    pending_trial.set_evaluating(time_started=0.0, worker_id="dummy")
    neps_state.put_updated_trial(pending_trial)

    def eval_function(*args, **kwargs) -> float:
        return 1.0

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    worker.run()

    assert worker.worker_cumulative_eval_count == 1
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0

    trials = neps_state.lock_and_read_trials()
    assert len(trials) == 2

    the_pending_trial = trials[pending_trial.id]
    assert the_pending_trial == pending_trial
    assert the_pending_trial.metadata.state == Trial.State.EVALUATING
    assert the_pending_trial.report is None

    the_completed_trial_id = next(iter(trials.keys() - {pending_trial.id}))
    the_completed_trial = trials[the_completed_trial_id]

    assert the_completed_trial.metadata.state == Trial.State.SUCCESS
    assert the_completed_trial.report is not None
    assert the_completed_trial.report.objective_to_minimize == 1.0


def test_max_cost_total(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        max_evaluations_total=10,  # Safety incase it doesn't work that we eventually stop
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=2,  # <- Highlight, only 2 maximum evaluations allowed
        max_evaluations_for_worker=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        return {"objective_to_minimize": 1.0, "cost": 1.0}

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    worker.run()

    assert worker.worker_cumulative_eval_count == 2
    assert worker.worker_cumulative_eval_cost == 2.0
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0

    trials = neps_state.lock_and_read_trials()
    assert len(trials) == 2

    # New worker should now not run anything as the total cost has been reached.
    new_worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    new_worker.run()
    assert new_worker.worker_cumulative_eval_count == 0


def test_worker_cost_total(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        max_evaluations_total=10,  # Safety incase it doesn't work that we eventually stop
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=2,  # <- Highlight, only 2 maximum evaluations allowed
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        return {"objective_to_minimize": 1.0, "cost": 1.0}

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    worker.run()

    assert worker.worker_cumulative_eval_count == 2
    assert worker.worker_cumulative_eval_cost == 2.0
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0

    trials = neps_state.lock_and_read_trials()
    assert len(trials) == 2

    # New worker should also run 2 more trials
    new_worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    new_worker.run()
    assert new_worker.worker_cumulative_eval_count == 2
    assert new_worker.worker_cumulative_eval_cost == 2.0
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0

    trials = neps_state.lock_and_read_trials()
    assert len(trials) == 4  # 2 more trials were ran


def test_worker_wallclock_time(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        max_evaluations_total=1000,  # Safety incase it doesn't work that we eventually stop
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=1,  # <- highlight, 1 second
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> float:
        return 1.0

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
        worker_id="dummy",
    )
    worker.run()

    assert worker.worker_cumulative_eval_count > 0
    assert worker.worker_cumulative_evaluation_time_seconds <= 2.0
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0
    len_trials_on_first_worker = len(neps_state.lock_and_read_trials())

    # New worker should also run some trials more trials
    new_worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
        worker_id="dummy2",
    )
    new_worker.run()
    assert new_worker.worker_cumulative_eval_count > 0
    assert new_worker.worker_cumulative_evaluation_time_seconds <= 2.0
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0
    len_trials_on_second_worker = len(neps_state.lock_and_read_trials())
    assert len_trials_on_second_worker > len_trials_on_first_worker


def test_max_worker_evaluation_time(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        max_evaluations_total=10,  # Safety incase it doesn't work that we eventually stop
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=0.5,
        max_cost_for_worker=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> float:
        time.sleep(0.6)
        return 1.0

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
        worker_id="dummy",
    )
    worker.run()

    assert worker.worker_cumulative_eval_count > 0
    assert worker.worker_cumulative_evaluation_time_seconds <= 1.0
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0
    len_trials_on_first_worker = len(neps_state.lock_and_read_trials())

    # New worker should also run some trials more trials
    new_worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
        worker_id="dummy2",
    )
    new_worker.run()
    assert new_worker.worker_cumulative_eval_count > 0
    assert new_worker.worker_cumulative_evaluation_time_seconds <= 1.0
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0
    len_trials_on_second_worker = len(neps_state.lock_and_read_trials())
    assert len_trials_on_second_worker > len_trials_on_first_worker


def test_max_evaluation_time_global(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        max_evaluations_total=10,  # Safety incase it doesn't work that we eventually stop
        include_in_progress_evaluations_towards_maximum=False,
        max_cost_total=None,
        max_evaluations_for_worker=None,
        max_evaluation_time_total_seconds=0.5,  # <- Highlight
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> float:
        time.sleep(0.6)
        return 1.0

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
        worker_id="dummy",
    )
    worker.run()

    assert worker.worker_cumulative_eval_count > 0
    assert worker.worker_cumulative_evaluation_time_seconds <= 1.0
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0
    len_trials_on_first_worker = len(neps_state.lock_and_read_trials())

    # New worker should also run some trials more trials
    new_worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
        worker_id="dummy2",
    )
    new_worker.run()
    assert new_worker.worker_cumulative_eval_count == 0
    assert new_worker.worker_cumulative_evaluation_time_seconds == 0
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0
    len_trials_on_second_worker = len(neps_state.lock_and_read_trials())
    assert len_trials_on_second_worker == len_trials_on_first_worker
