from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pytest_cases import fixture

import scalefm
from scalefm import Float, PipelineSpace, analyze, save_pipeline_results
from scalefm.optimizers import OptimizerInfo
from scalefm.optimizers.algorithms import random_search
from scalefm.runtime import DefaultWorker
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
    return ScaleFMState.create_or_load(
        path=tmp_path / "scalefm_state",
        optimizer_info=OptimizerInfo(name="blah", info={"nothing": "here"}),
        optimizer_state=OptimizationState(
            budget=None, seed_snapshot=SeedSnapshot.new_capture(), shared_state={}
        ),
        pipeline_space=ASpace(),
    )


class ASpace(PipelineSpace):
    a = Float(0, 1)


def test_async_happy_path_changes_state(scalefm_state: ScaleFMState) -> None:
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
                root_directory=Path(scalefm_state.path),
            )

        callback_holder.append(async_save)

    DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=async_eval_fn,
        settings=settings,
    ).run()

    trials = scalefm_state.lock_and_read_trials()
    assert len(trials) == 2
    for trial in trials.values():
        assert trial.metadata.state == Trial.State.EVALUATING
        assert trial.report is None

    result_dict = {"objective_to_minimize": 0.3, "cost": 1.2}
    callback_holder[0](result_dict)
    trials = scalefm_state.lock_and_read_trials()
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
    trial_after = list(scalefm_state.lock_and_read_trials().values())[1]
    assert trial_after.metadata.state == Trial.State.EVALUATING

    callback_holder[1]({"objective_to_minimize": 10, "cost": 2.0})
    trials = scalefm_state.lock_and_read_trials()
    trial_after = list(trials.values())[1]
    assert trial_after.metadata.state == Trial.State.SUCCESS
    assert trial_after.report.objective_to_minimize == 10
    assert trial_after.report.cost == 2.0


def test_async_save_updates_the_whole_summary(scalefm_state: ScaleFMState) -> None:
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
                root_directory=Path(scalefm_state.path),
            )

        callback_holder.append(async_save)

    DefaultWorker.new(
        state=scalefm_state,
        optimizer=optimizer,
        evaluation_fn=async_eval_fn,
        settings=settings,
    ).run()

    summary_dir = Path(scalefm_state.path) / "summary"

    callback_holder[0]({"objective_to_minimize": 0.3, "cost": 1.2})

    full = (summary_dir / "full.csv").read_text()
    assert "objective_to_minimize" in full
    assert "0.3" in full


def test_returning_none_without_async_raises(
    scalefm_state: ScaleFMState,
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
        state=scalefm_state,
        optimizer=random_search(ASpace()),
        evaluation_fn=lambda *_, **__: None,
        settings=settings,
    )
    with pytest.raises(ValueError, match="returned None"):
        worker.run()


def test_analyze_rebuilds_the_summary(scalefm_state: ScaleFMState) -> None:
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
        state=scalefm_state,
        optimizer=random_search(ASpace()),
        evaluation_fn=lambda *_, **__: {"objective_to_minimize": 0.5, "cost": 1.0},
        settings=settings,
    ).run()

    summary_dir = Path(scalefm_state.path) / "summary"
    shutil.rmtree(summary_dir)

    analyze(scalefm_state.path)

    assert "objective_to_minimize" in (summary_dir / "full.csv").read_text()
    assert list(summary_dir.rglob("*.png"))


def test_analyze_missing_directory_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        analyze(missing)
    assert not missing.exists()


def test_async_acquire_refuses_an_evaluate_pipeline(tmp_path: Path) -> None:
    """With `is_async=True`, `acquire` only samples, so it takes no function."""
    with pytest.raises(ValueError, match="must be None"):
        scalefm.acquire(
            evaluate_pipeline=lambda **_: None,
            pipeline_space=ASpace(),
            root_directory=tmp_path / "root",
            is_async=True,
            worker_evaluations_to_spend=1,
        )
    assert not (tmp_path / "root").exists()


def test_sync_acquire_requires_an_evaluate_pipeline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is required"):
        scalefm.acquire(
            pipeline_space=ASpace(),
            root_directory=tmp_path / "root",
            worker_evaluations_to_spend=1,
        )


def test_async_acquire_samples_configs_a_job_reports_later(tmp_path: Path) -> None:
    """`acquire(is_async=True)` leaves its configs pending in `config.yaml`, and
    `save_pipeline_results` completes them.
    """
    root = tmp_path / "root"
    scalefm.acquire(
        pipeline_space=ASpace(),
        root_directory=root,
        is_async=True,
        worker_evaluations_to_spend=2,
    )

    config_dirs = sorted((root / "configs").glob("config_*"))
    assert len(config_dirs) == 2
    assert all((d / "config.yaml").exists() for d in config_dirs)
    assert not any((d / "report.yaml").exists() for d in config_dirs)

    pipeline_id = config_dirs[0].name.removeprefix("config_")
    save_pipeline_results(
        user_result={"objective_to_minimize": 0.3, "cost": 1.0},
        pipeline_id=pipeline_id,
        root_directory=root,
    )
    states = {
        trial.id: trial.metadata.state
        for trial in ScaleFMState.create_or_load(root, load_only=True)
        .lock_and_read_trials()
        .values()
    }
    assert states[pipeline_id] == Trial.State.SUCCESS
    assert list(states.values()).count(Trial.State.EVALUATING) == 1


def test_acquire_defaults_to_random_search(tmp_path: Path) -> None:
    root = tmp_path / "root"
    scalefm.acquire(
        evaluate_pipeline=lambda a: {"objective_to_minimize": a, "cost": 1.0},
        pipeline_space=ASpace(),
        root_directory=root,
        worker_evaluations_to_spend=1,
    )
    assert scalefm.load_optimizer_info(root)["name"] == "random_search"
