from __future__ import annotations

from pathlib import Path

import pytest
from pytest_cases import fixture

from neps import Float, PipelineSpace, analyze, save_pipeline_results
from neps.optimizers import OptimizerInfo
from neps.optimizers.algorithms import random_search
from neps.runtime import DefaultWorker
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
        pipeline_space=ASpace(),
    )


class ASpace(PipelineSpace):
    a = Float(0, 1)


def test_async_happy_path_changes_state(neps_state: NePSState) -> None:
    optimizer = random_search(ASpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(
            cost_if_not_provided=10
        ),  # <- it is ignored
        evaluations_to_spend=2,
        include_in_progress_evaluations_towards_maximum=True,
        cost_to_spend=1,
        fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
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
    trials = neps_state.lock_and_read_trials()
    trial_iter = iter(trials.values())
    trial_after = next(trial_iter)
    assert trial_after.metadata.state == Trial.State.SUCCESS
    assert trial_after.report.objective_to_minimize == 0.3
    assert trial_after.report.cost == 1.2

    # second trial is not submitted yet
    trial_after = next(trial_iter)
    assert trial_after.metadata.state == Trial.State.EVALUATING

    result_dict = {"objective_to_minimize": 10}  # cost not provided
    callback_holder[1](result_dict)
    trials = neps_state.lock_and_read_trials()
    trial_after = list(trials.values())[1]
    assert trial_after.metadata.state == Trial.State.SUCCESS
    assert trial_after.report.objective_to_minimize == 10
    assert trial_after.report.cost is None  # default is always None


def test_async_save_updates_the_whole_summary(neps_state: NePSState) -> None:
    """The detached path writes the same summary artifacts as the worker loop,
    not only the CSVs.
    """
    optimizer = random_search(ASpace())
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        evaluations_to_spend=1,
        include_in_progress_evaluations_towards_maximum=True,
        cost_to_spend=None,
        fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    callback_holder: list[callable] = []

    def async_eval_fn(*_, pipeline_id, pipeline_directory, **__):
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

    summary_dir = Path(neps_state.path) / "summary"
    assert (summary_dir / "best_config.txt").read_text() == ""

    callback_holder[0]({"objective_to_minimize": 0.3, "cost": 1.2})

    assert "Objective to minimize: 0.3" in (summary_dir / "best_config.txt").read_text()
    assert "Config ID:" in (summary_dir / "best_config_trajectory.txt").read_text()
    assert "objective_to_minimize" in (summary_dir / "full.csv").read_text()
    assert "best_objective_to_minimize" in (summary_dir / "short.csv").read_text()


def test_analyze_rebuilds_the_summary(neps_state: NePSState) -> None:
    """`analyze` rebuilds the whole summary folder, plots included, from disk."""
    settings = WorkerSettings(
        on_error=OnErrorPossibilities.IGNORE,
        default_report_values=DefaultReportValues(),
        evaluations_to_spend=3,
        include_in_progress_evaluations_towards_maximum=True,
        cost_to_spend=None,
        fidelities_to_spend=None,
        max_evaluation_time_total_seconds=None,
        max_wallclock_time_seconds=None,
        batch_size=None,
    )

    DefaultWorker.new(
        state=neps_state,
        optimizer=random_search(ASpace()),
        evaluation_fn=lambda *_, **__: 0.5,
        settings=settings,
    ).run()

    summary_dir = Path(neps_state.path) / "summary"
    for file in summary_dir.iterdir():
        file.unlink()

    analyze(neps_state.path)

    assert "Config ID:" in (summary_dir / "best_config.txt").read_text()
    assert "Final cumulative metrics" in (summary_dir / "best_config.txt").read_text()
    assert "Config ID:" in (summary_dir / "best_config_trajectory.txt").read_text()
    assert (summary_dir / "full.csv").exists()
    assert (summary_dir / "short.csv").exists()
    assert (summary_dir / "incumbent_trajectory.csv").exists()
    assert list(summary_dir.glob("incumbent_trajectory.*")) != [
        summary_dir / "incumbent_trajectory.csv"
    ]


def test_analyze_missing_directory_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        analyze(missing)
    assert not missing.exists()
