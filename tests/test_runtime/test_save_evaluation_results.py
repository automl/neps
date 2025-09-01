from __future__ import annotations

from pathlib import Path

from pytest_cases import fixture

from neps import save_pipeline_results
from neps.optimizers import OptimizerInfo
from neps.optimizers.algorithms import random_search
from neps.runtime import DefaultWorker
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
            budget=None, seed_snapshot=SeedSnapshot.new_capture(), shared_state={}
        ),
    )


def test_async_happy_path_changes_state(neps_state: NePSState) -> None:
    optimizer = random_search(SearchSpace({"a": Float(0, 1)}))
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(
            cost_if_not_provided=1.2,
        ),
        cost_to_spend=None,
        max_evaluations_for_worker=2,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_for_worker_seconds=None,
        max_evaluation_time_for_worker_seconds=None,
        max_cost_for_worker=None,
        batch_size=None,
        fidelities_to_spend=None,
        evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=False,
    )

    callback_holder: list[callable] = []

    def async_eval_fn(*_, pipeline_id, pipeline_directory, **__):
        # run async after 5 seconds
        def async_save(user_result):
            save_pipeline_results(
                pipeline_id=pipeline_id,
                user_result=user_result,
                root_directory=Path(neps_state.path),
            )

        callback_holder.append(async_save)

    DefaultWorker.new(
        state=neps_state,
        optimizer=optimizer,
        evaluation_fn=async_eval_fn,
        settings=settings,
    ).run()

    trials = neps_state.lock_and_read_trials()
    assert len(trials) == 2
    for trial in trials.values():
        assert trial.metadata.state == Trial.State.EVALUATING
        assert trial.report is None

    result_dict = {"objective_to_minimize": 0.3, "cost": 1.2}
    callback_holder[0](result_dict)
    trial_iter = iter(neps_state.lock_and_read_trials().values())
    trial_after = next(trial_iter)
    assert trial_after.metadata.state == Trial.State.SUCCESS
    assert trial_after.report.objective_to_minimize == 0.3
    assert trial_after.report.cost == 1.2

    # second trial is not submitted yet
    trial_after = next(trial_iter)
    assert trial_after.metadata.state == Trial.State.EVALUATING
