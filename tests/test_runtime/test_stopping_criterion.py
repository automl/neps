from __future__ import annotations

import time
from pathlib import Path

from pytest_cases import fixture

from neps.optimizers.algorithms import asha, random_search
from neps.optimizers.optimizer import OptimizerInfo
from neps.runtime import DefaultWorker
from neps.space import HPOFloat, HPOInteger, SearchSpace
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


def test_evaluations_to_spend_stopping_criterion(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace({"a": HPOFloat(0, 1)}))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        evaluations_to_spend=3,  # <- Highlight
        include_in_progress_evaluations_towards_maximum=False,
        cost_to_spend=None,  # <- cost to spend will not be checked
        fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> float:
        return {"objective_to_minimize": 1.0, "cost": 3.0}

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    worker.run()

    trials = list(neps_state.lock_and_read_trials().values())

    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        == 3
    )
    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 0

    trials = neps_state.lock_and_read_trials()
    for _, trial in trials.items():
        assert trial.metadata.state == Trial.State.SUCCESS
        assert trial.report is not None
        assert trial.report.objective_to_minimize == 1.0
        assert trial.report.cost == 3.0

    # New worker also runs for another 3 evaluations
    new_worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    new_worker.run()

    assert (
        sum(
            1
            for trial in list(neps_state.lock_and_read_trials().values())
            if trial.metadata.evaluating_worker_id == new_worker.worker_id
        )
        == 3
    )
    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 0


def test_multiple_criteria_set(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace({"a": HPOFloat(0, 1)}))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        evaluations_to_spend=5,  # <- Highlight
        include_in_progress_evaluations_towards_maximum=False,
        cost_to_spend=3,  # <- Highlight
        fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        return {"objective_to_minimize": 1.0, "cost": 2.0}

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    worker.run()

    # The cost_to_spend criterion should stop the worker first after 3 evaluations
    trials = list(neps_state.lock_and_read_trials().values())
    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        == 2
    )  # til the cost_to_spend is reached
    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 0

    assert len(trials) == 2
    for trial in trials:
        assert trial.metadata.state == Trial.State.SUCCESS
        assert trial.report is not None
        assert trial.report.objective_to_minimize == 1.0
        assert trial.report.cost == 2.0

    settings.cost_to_spend = 100  # now only evaluations_to_spend matters
    new_worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    new_worker.run()
    trials = list(neps_state.lock_and_read_trials().values())
    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == new_worker.worker_id
        )
        == 5
    )
    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 0


def test_include_in_progress_evaluations_towards_maximum_with_work_eval_count(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace({"a": HPOFloat(0, 1)}))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        evaluations_to_spend=2,  # <- Highlight, only 2 maximum evaluations allowed
        include_in_progress_evaluations_towards_maximum=True,  # <- inprogress trial
        cost_to_spend=None,
        fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> float:
        return 1.0

    worker = DefaultWorker.new(
        worker_id="test",
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )

    # We put in one trial as being inprogress
    pending_trial = neps_state.lock_and_sample_trial(
        optimizer, worker_id=worker.worker_id
    )
    pending_trial.set_evaluating(time_started=0.0, worker_id=worker.worker_id)
    neps_state.put_updated_trial(pending_trial)

    worker.run()

    trials_dict = neps_state.lock_and_read_trials()
    trials = list(trials_dict.values())
    assert len(trials) == 2  # only one more trial should have been evaluated
    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        == 2
    )
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0

    the_pending_trial = trials_dict[pending_trial.id]
    assert the_pending_trial == pending_trial
    assert the_pending_trial.metadata.state == Trial.State.EVALUATING
    assert the_pending_trial.report is None

    the_completed_trial_id = next(iter(trials_dict.keys() - {pending_trial.id}))
    the_completed_trial = trials_dict[the_completed_trial_id]

    assert the_completed_trial.metadata.state == Trial.State.SUCCESS
    assert the_completed_trial.report is not None
    assert the_completed_trial.report.objective_to_minimize == 1.0


def test_worker_wallclock_time(neps_state: NePSState) -> None:
    optimizer = random_search(pipeline_space=SearchSpace({"a": HPOFloat(0, 1)}))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        evaluations_to_spend=1000,  # Incase it doesn't work that we eventually stop
        include_in_progress_evaluations_towards_maximum=False,
        cost_to_spend=None,
        fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=1,  # <- highlight, 1 second
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> float:
        return 1.0

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        worker_id="dummy",
    )
    time_start = time.monotonic()
    worker.run()
    time_end = time.monotonic()

    trials = list(neps_state.lock_and_read_trials().values())
    assert len(trials) > 0  # should have done some evaluations
    assert time_end - time_start >= 1.0  # should have run for at least 1 second
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0
    len(neps_state.lock_and_read_trials())


def test_max_worker_evaluation_time(neps_state: NePSState) -> None:
    optimizer = random_search(pipeline_space=SearchSpace({"a": HPOFloat(0, 1)}))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        evaluations_to_spend=100,  # Safety incase it doesn't work that we eventually stop
        include_in_progress_evaluations_towards_maximum=False,
        cost_to_spend=None,
        fidelities_to_spend=None,
        max_evaluation_time_total_seconds=0.5,
        max_wallclock_time_seconds=None,
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
        worker_id="dummy",
    )
    worker.run()

    trials = neps_state.lock_and_read_trials().values()
    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        > 0
    )
    assert (
        0.5
        <= sum(
            trial.metadata.evaluation_duration
            for trial in trials
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        <= 1.0
    )  # some margin for time spent for the last evaluation that went over the limit.

    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0


def test_fidelity_to_spend(neps_state: NePSState) -> None:
    optimizer = asha(
        pipeline_space=SearchSpace(
            {"a": HPOFloat(0, 1), "b": HPOInteger(2, 10, is_fidelity=True)}
        )
    )
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        evaluations_to_spend=10,  # Safety incase it doesn't work that we eventually stop
        include_in_progress_evaluations_towards_maximum=False,
        cost_to_spend=None,
        fidelities_to_spend=2,  # this is the min fidelity used in an evaluation
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
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
        worker_id="dummy",
    )
    worker.run()

    trials1 = list(neps_state.lock_and_read_trials().values())
    assert (
        sum(
            1
            for trial in trials1
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        == 1
    )

    assert (
        10
        >= sum(
            trial.config.get("b")
            for trial in trials1
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        >= 2
    )

    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0

    # New worker should also run some trials more trials
    settings.fidelities_to_spend = 4  # now only 2 fidelity matters
    new_worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        worker_id="dummy2",
    )
    new_worker.run()
    trials2 = list(neps_state.lock_and_read_trials().values())
    assert (
        13
        >= sum(
            trial.config.get("b")
            for trial in trials2
            if trial.metadata.evaluating_worker_id == new_worker.worker_id
        )
        >= 4
    )
    assert (
        neps_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(neps_state.lock_and_get_errors()) == 0
