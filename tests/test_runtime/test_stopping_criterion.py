from __future__ import annotations

import time
from pathlib import Path

from pytest_cases import fixture

from scalefm.exceptions import BudgetExhaustedError
from scalefm.optimizers.algorithms import random_search
from scalefm.optimizers.optimizer import OptimizerInfo
from scalefm.runtime import DefaultWorker
from scalefm.space.scalefm_spaces.parameters import Float, PipelineSpace
from scalefm.state import (
    DefaultReportValues,
    OnErrorPossibilities,
    OptimizationState,
    ScaleFMState,
    SeedSnapshot,
    Trial,
    WorkerSettings,
)


@fixture
def scalefm_state(tmp_path: Path) -> ScaleFMState:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    return ScaleFMState.create_or_load(
        path=tmp_path / "scalefm_state",
        optimizer_info=OptimizerInfo(name="blah", info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None,
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state=None,
        ),
        pipeline_space=TestSpace(),
    )


def test_evaluations_to_spend_stopping_criterion(
    scalefm_state: ScaleFMState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(pipeline_space=TestSpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=3,  # <- Highlight
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,  # <- cost to spend will not be checked
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        return {"objective_to_minimize": 1.0, "cost": 3.0}

    worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    worker.run()

    trials = list(scalefm_state.lock_and_read_trials().values())

    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        == 3
    )
    assert scalefm_state.lock_and_get_next_pending_trial() is None
    assert len(scalefm_state.lock_and_get_errors()) == 0

    trials = scalefm_state.lock_and_read_trials()
    for _, trial in trials.items():
        assert trial.metadata.state == Trial.State.SUCCESS
        assert trial.report is not None
        assert trial.report.objective_to_minimize == 1.0
        assert trial.report.cost == 3.0

    # New worker also runs for another 3 evaluations
    new_worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    new_worker.run()

    assert (
        sum(
            1
            for trial in list(scalefm_state.lock_and_read_trials().values())
            if trial.metadata.evaluating_worker_id == new_worker.worker_id
        )
        == 3
    )
    assert scalefm_state.lock_and_get_next_pending_trial() is None
    assert len(scalefm_state.lock_and_get_errors()) == 0


def test_total_evaluations_to_spend_stopping_criterion(
    scalefm_state: ScaleFMState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(pipeline_space=TestSpace())
    scalefm_state.lock_and_update_global_budgets(
        total_evaluations_to_spend=5,
        total_cost_to_spend=None,
    )
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=3,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        return {"objective_to_minimize": 1.0, "cost": 1.0}

    worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    worker.run()

    trials = list(scalefm_state.lock_and_read_trials().values())

    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        == 3
    )

    # Raise the local limit as a safety net so the global limit stops this worker first.
    settings.worker_evaluations_to_spend = 10

    new_worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    new_worker.run()

    trials = list(scalefm_state.lock_and_read_trials().values())

    assert len(trials) == 5
    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == new_worker.worker_id
        )
        == 2
    )
    assert len(scalefm_state.lock_and_get_errors()) == 0


def test_total_cost_to_spend_stopping_criterion(
    scalefm_state: ScaleFMState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(pipeline_space=TestSpace())
    scalefm_state.lock_and_update_global_budgets(
        total_evaluations_to_spend=None,
        total_cost_to_spend=6,
    )
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=2,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        return {"objective_to_minimize": 1.0, "cost": 2.0}

    worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    worker.run()

    trials = list(scalefm_state.lock_and_read_trials().values())

    assert len(trials) == 2
    assert (
        sum(
            trial.report.cost
            for trial in trials
            if trial.report is not None and trial.report.cost is not None
        )
        == 4
    )

    settings.worker_evaluations_to_spend = 10
    new_worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    new_worker.run()

    trials = list(scalefm_state.lock_and_read_trials().values())

    assert len(trials) == 3
    assert (
        sum(
            trial.report.cost
            for trial in trials
            if trial.report is not None and trial.report.cost is not None
        )
        == 6
    )
    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == new_worker.worker_id
        )
        == 1
    )
    assert len(scalefm_state.lock_and_get_errors()) == 0


def test_global_budget_can_be_updated(scalefm_state: ScaleFMState) -> None:
    scalefm_state.lock_and_update_global_budgets(
        total_evaluations_to_spend=5,
        total_cost_to_spend=None,
    )

    optimizer_state = scalefm_state.lock_and_get_optimizer_state()
    assert optimizer_state.budget is not None
    assert optimizer_state.budget.total_evaluations_to_spend == 5

    scalefm_state.lock_and_update_global_budgets(
        total_evaluations_to_spend=10,
        total_cost_to_spend=None,
    )

    optimizer_state = scalefm_state.lock_and_get_optimizer_state()
    assert optimizer_state.budget is not None
    assert optimizer_state.budget.total_evaluations_to_spend == 10

    # Passing None means "do not change the stored global budget".
    scalefm_state.lock_and_update_global_budgets(
        total_evaluations_to_spend=None,
        total_cost_to_spend=None,
    )

    optimizer_state = scalefm_state.lock_and_get_optimizer_state()
    assert optimizer_state.budget is not None
    assert optimizer_state.budget.total_evaluations_to_spend == 10


def eval_function(*_args, **_kwargs) -> dict:
    return {"objective_to_minimize": 1.0, "cost": 1.0}


def test_worker_uses_updated_global_budget(
    scalefm_state: ScaleFMState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(pipeline_space=TestSpace())

    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=10,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    scalefm_state.lock_and_update_global_budgets(
        total_evaluations_to_spend=5,
        total_cost_to_spend=None,
    )

    worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )

    # Change the shared budget after the worker has already been created.
    scalefm_state.lock_and_update_global_budgets(
        total_evaluations_to_spend=1,
        total_cost_to_spend=None,
    )

    worker.run()

    trials = list(scalefm_state.lock_and_read_trials().values())

    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        == 1
    )


def test_multiple_criteria_set(
    scalefm_state: ScaleFMState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(pipeline_space=TestSpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=5,  # <- Highlight
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=3,  # <- Highlight
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        return {"objective_to_minimize": 1.0, "cost": 2.0}

    worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    worker.run()

    # The worker_cost_to_spend criterion should stop the worker first after 3 evaluations
    trials = list(scalefm_state.lock_and_read_trials().values())
    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == worker.worker_id
        )
        == 2
    )  # til the worker_cost_to_spend is reached
    assert scalefm_state.lock_and_get_next_pending_trial() is None
    assert len(scalefm_state.lock_and_get_errors()) == 0

    assert len(trials) == 2
    for trial in trials:
        assert trial.metadata.state == Trial.State.SUCCESS
        assert trial.report is not None
        assert trial.report.objective_to_minimize == 1.0
        assert trial.report.cost == 2.0

    settings.worker_cost_to_spend = 100  # now only worker_evaluations_to_spend matters
    new_worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    new_worker.run()
    trials = list(scalefm_state.lock_and_read_trials().values())
    assert (
        sum(
            1
            for trial in trials
            if trial.metadata.evaluating_worker_id == new_worker.worker_id
        )
        == 5
    )
    assert scalefm_state.lock_and_get_next_pending_trial() is None
    assert len(scalefm_state.lock_and_get_errors()) == 0


def test_include_in_progress_evaluations_towards_maximum_with_work_eval_count(
    scalefm_state: ScaleFMState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(pipeline_space=TestSpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=2,  # <- Highlight, only 2 maximum evaluations allowed
        include_in_progress_evaluations_towards_maximum=True,  # <- inprogress trial
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        return {"objective_to_minimize": 1.0, "cost": 1.0}

    worker = DefaultWorker.new(
        worker_id="test",
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )

    # We put in one trial as being inprogress
    pending_trial = scalefm_state.lock_and_sample_trial(
        optimizer, worker_id=worker.worker_id
    )
    pending_trial.set_evaluating(time_started=0.0, worker_id=worker.worker_id)
    scalefm_state.put_updated_trial(pending_trial)

    worker.run()

    trials_dict = scalefm_state.lock_and_read_trials()
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
        scalefm_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(scalefm_state.lock_and_get_errors()) == 0

    the_pending_trial = trials_dict[pending_trial.id]
    assert the_pending_trial == pending_trial
    assert the_pending_trial.metadata.state == Trial.State.EVALUATING
    assert the_pending_trial.report is None

    the_completed_trial_id = next(iter(trials_dict.keys() - {pending_trial.id}))
    the_completed_trial = trials_dict[the_completed_trial_id]

    assert the_completed_trial.metadata.state == Trial.State.SUCCESS
    assert the_completed_trial.report is not None
    assert the_completed_trial.report.objective_to_minimize == 1.0


def test_worker_wallclock_time(scalefm_state: ScaleFMState) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(pipeline_space=TestSpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1000,  # Safety limit
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=1,  # <- highlight, 1 second
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        return {"objective_to_minimize": 1.0, "cost": 1.0}

    worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        worker_id="dummy",
    )
    time_start = time.monotonic()
    worker.run()
    time_end = time.monotonic()

    trials = list(scalefm_state.lock_and_read_trials().values())
    assert len(trials) > 0  # should have done some evaluations
    assert time_end - time_start >= 1.0  # should have run for at least 1 second
    assert (
        scalefm_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(scalefm_state.lock_and_get_errors()) == 0
    len(scalefm_state.lock_and_read_trials())


def test_max_worker_evaluation_time(scalefm_state: ScaleFMState) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer = random_search(pipeline_space=TestSpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=100,  # Safety limit
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=0.5,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        time.sleep(0.6)
        return {"objective_to_minimize": 1.0, "cost": 1.0}

    worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        worker_id="dummy",
    )
    worker.run()

    trials = scalefm_state.lock_and_read_trials().values()
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
        scalefm_state.lock_and_get_next_pending_trial() is None
    )  # should have no pending trials to be picked up
    assert len(scalefm_state.lock_and_get_errors()) == 0


def test_budget_exhausted_optimizer_stops_the_worker_without_retrying(
    scalefm_state: ScaleFMState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    sample = random_search(pipeline_space=TestSpace())
    calls = 0

    def optimizer(trials, budget_info, n=None):
        nonlocal calls
        calls += 1
        if calls > 2:
            raise BudgetExhaustedError("No budget left")
        return sample(trials, budget_info, n=n)

    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=None,
        include_in_progress_evaluations_towards_maximum=False,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    def eval_function(*args, **kwargs) -> dict:
        return {"objective_to_minimize": 1.0, "cost": 1.0}

    worker = DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
    )
    worker.run()

    assert calls == 3
    assert len(scalefm_state.lock_and_read_trials()) == 2
