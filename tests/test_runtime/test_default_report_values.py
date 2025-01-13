from __future__ import annotations

from pathlib import Path

from pytest_cases import fixture

from neps.optimizers import random_search
from neps.runtime import DefaultWorker
from neps.search_spaces import Float, SearchSpace
from neps.state.neps_state import NePSState
from neps.state.optimizer import OptimizationState, OptimizerInfo
from neps.state.seed_snapshot import SeedSnapshot
from neps.state.settings import DefaultReportValues, OnErrorPossibilities, WorkerSettings
from neps.state.trial import Trial


@fixture
def neps_state(tmp_path: Path) -> NePSState:
    return NePSState.create_or_load(
        path=tmp_path / "neps_state",
        optimizer_info=OptimizerInfo(info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None, seed_snapshot=SeedSnapshot.new_capture(), shared_state={}
        ),
    )


def test_default_values_on_error(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(pipeline_space=SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(
            objective_value_on_error=2.4,  # <- Highlight
            cost_value_on_error=2.4,  # <- Highlight
            learning_curve_on_error=[2.4, 2.5],  # <- Highlight
        ),
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
        _pre_sample_hooks=None,
    )
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

    trial = trials.popitem()[1]
    assert trial.metadata.state == Trial.State.CRASHED
    assert trial.report is not None
    assert trial.report.objective_to_minimize == 2.4
    assert trial.report.cost == 2.4
    assert trial.report.learning_curve == [2.4, 2.5]


def test_default_values_on_not_specified(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(
            cost_if_not_provided=2.4,
            learning_curve_if_not_provided=[2.4, 2.5],
        ),
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
        return 1.0

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    worker.run()

    trials = neps_state.lock_and_read_trials()
    n_sucess = sum(
        trial.metadata.state == Trial.State.SUCCESS is not None
        for trial in trials.values()
    )
    assert len(trials) == 1
    assert n_sucess == 1

    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 0

    trial = trials.popitem()[1]
    assert trial.metadata.state == Trial.State.SUCCESS
    assert trial.report is not None
    assert trial.report.cost == 2.4
    assert trial.report.learning_curve == [2.4, 2.5]


def test_default_value_objective_to_minimize_curve_take_objective_to_minimize_value(
    neps_state: NePSState,
) -> None:
    optimizer = random_search(SearchSpace(a=Float(0, 1)))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(
            learning_curve_if_not_provided="objective_to_minimize"
        ),
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

    LOSS = 1.0

    def eval_function(*args, **kwargs) -> float:
        return LOSS

    worker = DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=eval_function,
        settings=settings,
        _pre_sample_hooks=None,
    )
    worker.run()

    trials = neps_state.lock_and_read_trials()
    n_sucess = sum(
        trial.metadata.state == Trial.State.SUCCESS is not None
        for trial in trials.values()
    )
    assert len(trials) == 1
    assert n_sucess == 1

    assert neps_state.lock_and_get_next_pending_trial() is None
    assert len(neps_state.lock_and_get_errors()) == 0

    trial = trials.popitem()[1]
    assert trial.metadata.state == Trial.State.SUCCESS
    assert trial.report is not None
    assert trial.report.learning_curve == [LOSS]
