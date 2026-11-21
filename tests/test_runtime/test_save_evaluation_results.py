from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pytest_cases import fixture

from extrapolaw import Float, PipelineSpace, analyze, save_pipeline_results
from extrapolaw.optimizers import OptimizerInfo
from extrapolaw.optimizers.algorithms import random_search
from extrapolaw.runtime import DefaultWorker
from extrapolaw.state import (
    DefaultReportValues,
    ExtrapolawState,
    OnErrorPossibilities,
    OptimizationState,
    SeedSnapshot,
    Trial,
    WorkerSettings,
)


@fixture
def extrapolaw_state(tmp_path: Path) -> ExtrapolawState:
    return ExtrapolawState.create_or_load(
        path=tmp_path / "extrapolaw_state",
        optimizer_info=OptimizerInfo(name="blah", info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None, seed_snapshot=SeedSnapshot.new_capture(), shared_state={}
        ),
        pipeline_space=ASpace(),
    )


class ASpace(PipelineSpace):
    a = Float(0, 1)


def test_async_happy_path_changes_state(extrapolaw_state: ExtrapolawState) -> None:
    optimizer = random_search(ASpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(
            cost_if_not_provided=10
        ),  # <- it is ignored
        worker_evaluations_to_spend=2,
        include_in_progress_evaluations_towards_maximum=True,
        worker_cost_to_spend=1,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
        is_async=True,
    )

    callback_holder: list[callable] = []

    def async_eval_fn(*_, pipeline_id, pipeline_directory, **__):
        # run async after 5 seconds
        def async_save(user_result):
            save_pipeline_results(
                pipeline_id=pipeline_id,
                user_result=user_result,
                root_directory=Path(extrapolaw_state.path),
            )

        callback_holder.append(async_save)

    DefaultWorker.new(
        state=extrapolaw_state,
        optimizer=optimizer,
        evaluation_fn=async_eval_fn,
        settings=settings,
    ).run()

    trials = extrapolaw_state.lock_and_read_trials()
    assert len(trials) == 2
    for trial in trials.values():
        assert trial.metadata.state == Trial.State.EVALUATING
        assert trial.report is None

    result_dict = {"objective_to_minimize": 0.3, "cost": 1.2}
    callback_holder[0](result_dict)
    trials = extrapolaw_state.lock_and_read_trials()
    trial_iter = iter(trials.values())
    trial_after = next(trial_iter)
    assert trial_after.metadata.state == Trial.State.SUCCESS
    assert trial_after.report.objective_to_minimize == 0.3
    assert trial_after.report.cost == 1.2

    # second trial is not submitted yet
    trial_after = next(trial_iter)
    assert trial_after.metadata.state == Trial.State.EVALUATING

    # Without a cost, the result is refused and the trial is still being evaluated
    with pytest.raises(ValueError, match="no 'cost'"):
        callback_holder[1]({"objective_to_minimize": 10})
    trial_after = list(extrapolaw_state.lock_and_read_trials().values())[1]
    assert trial_after.metadata.state == Trial.State.EVALUATING

    callback_holder[1]({"objective_to_minimize": 10, "cost": 2.0})
    trials = extrapolaw_state.lock_and_read_trials()
    trial_after = list(trials.values())[1]
    assert trial_after.metadata.state == Trial.State.SUCCESS
    assert trial_after.report.objective_to_minimize == 10
    assert trial_after.report.cost == 2.0


def test_async_save_updates_the_whole_summary(extrapolaw_state: ExtrapolawState) -> None:
    """The detached path writes the same summary artifacts as the worker loop,
    not only the CSVs.
    """
    optimizer = random_search(ASpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=True,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
        is_async=True,
    )

    callback_holder: list[callable] = []

    def async_eval_fn(*_, pipeline_id, pipeline_directory, **__):
        def async_save(user_result):
            save_pipeline_results(
                pipeline_id=pipeline_id,
                user_result=user_result,
                root_directory=Path(extrapolaw_state.path),
            )

        callback_holder.append(async_save)

    DefaultWorker.new(
        state=extrapolaw_state,
        optimizer=optimizer,
        evaluation_fn=async_eval_fn,
        settings=settings,
    ).run()

    summary_dir = Path(extrapolaw_state.path) / "summary"

    callback_holder[0]({"objective_to_minimize": 0.3, "cost": 1.2})

    full = (summary_dir / "full.csv").read_text()
    assert "objective_to_minimize" in full
    assert "0.3" in full


def test_returning_none_without_async_raises(
    extrapolaw_state: ExtrapolawState,
) -> None:
    """Returning None is only allowed when the worker runs in async mode."""
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=True,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    worker = DefaultWorker.new(
        state=extrapolaw_state,
        optimizer=random_search(ASpace()),
        evaluation_fn=lambda *_, **__: None,
        settings=settings,
    )
    with pytest.raises(ValueError, match="returned None"):
        worker.run()


def test_analyze_rebuilds_the_summary(extrapolaw_state: ExtrapolawState) -> None:
    """`analyze` rebuilds the whole summary folder, plots included, from disk."""
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        worker_evaluations_to_spend=3,
        include_in_progress_evaluations_towards_maximum=True,
        worker_cost_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    DefaultWorker.new(
        state=extrapolaw_state,
        optimizer=random_search(ASpace()),
        evaluation_fn=lambda *_, **__: {"objective_to_minimize": 0.5, "cost": 1.0},
        settings=settings,
    ).run()

    summary_dir = Path(extrapolaw_state.path) / "summary"
    shutil.rmtree(summary_dir)

    analyze(extrapolaw_state.path)

    assert "objective_to_minimize" in (summary_dir / "full.csv").read_text()
    assert list(summary_dir.rglob("*.png"))


def test_analyze_missing_directory_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        analyze(missing)
    assert not missing.exists()
