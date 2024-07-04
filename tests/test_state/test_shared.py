from __future__ import annotations

from pytest_cases import parametrize_with_cases, case
import copy
import numpy as np
import random
from neps.state.shared import Shared
from neps.state.trial import Trial
import pytest
from typing import Callable
from pathlib import Path
from neps.state import SeedSnapshot, Shared, Trial, JobQueue, EvaluateJob, SampleJob

@case
def case_trial_1(tmp_path: Path) -> tuple[Shared[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=0,
        previous_trial=None,
        worker_id=0
    )
    def _mutate(trial: Trial) -> None:
        trial.set_submitted(time_submitted=1)

    return trial.as_filesystem_shared(tmp_path / "1"), _mutate

@case
def case_trial_2(tmp_path: Path) -> tuple[Shared[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=0,
        previous_trial=None,
        worker_id=0
    )
    trial.set_submitted(time_submitted=1)
    def _mutate(trial: Trial) -> None:
        trial.set_in_progress(time_started=2, worker_id=1)

    return trial.as_filesystem_shared(tmp_path / "1"), _mutate

@case
def case_trial_3(tmp_path: Path) -> tuple[Shared[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=0,
        previous_trial=None,
        worker_id=0
    )
    trial.set_submitted(time_submitted=1)
    trial.set_in_progress(time_started=2, worker_id=1)
    def _mutate(trial: Trial) -> None:
        trial.set_success(time_end=3, loss=1, cost=1, extra={"hello": [1, 2, 3]}, account_for_cost=True)

    return trial.as_filesystem_shared(tmp_path / "1"), _mutate

@case
def case_trial_4(tmp_path: Path) -> tuple[Shared[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=0,
        previous_trial=None,
        worker_id=0
    )
    trial.set_submitted(time_submitted=1)
    trial.set_in_progress(time_started=2, worker_id=1)
    def _mutate(trial: Trial) -> None:
        trial.set_crashed(time_end=3, loss=np.nan, cost=np.inf, extra={"hello": [1, 2, 3]}, account_for_cost=False)

    return trial.as_filesystem_shared(tmp_path / "1"), _mutate

@case
def case_trial_5(tmp_path: Path) -> tuple[Shared[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=0,
        previous_trial=None,
        worker_id=0
    )
    trial.set_submitted(time_submitted=1)
    trial.set_in_progress(time_started=2, worker_id=1)
    def _mutate(trial: Trial) -> None:
        trial.set_corrupted()

    return trial.as_filesystem_shared(tmp_path / "1"), _mutate

@case
def case_trial_6(tmp_path: Path) -> tuple[Shared[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=0,
        previous_trial=None,
        worker_id=0
    )
    trial.set_submitted(time_submitted=1)
    trial.set_in_progress(time_started=2, worker_id=1)
    trial.set_crashed(time_end=3, loss=np.nan, cost=np.inf, extra={"hello": [1, 2, 3]}, account_for_cost=False)
    def _mutate(trial: Trial) -> None:
        trial.reset()

    return trial.as_filesystem_shared(tmp_path / "1"), _mutate

@case
def case_jobqueue_sample(tmp_path: Path) -> tuple[Shared[JobQueue, Path], Callable[[JobQueue], None]]:
    jobqueue = JobQueue()
    jobqueue.push(SampleJob.new(issued_by="worker_1"))

    def _mutate(jobqueue: JobQueue) -> None:
        jobqueue.pop()

    return jobqueue.as_filesystem_shared(tmp_path / "jobqueue"), _mutate

@case
def case_jobqueue_evaluate(tmp_path: Path) -> tuple[Shared[JobQueue, Path], Callable[[JobQueue], None]]:
    jobqueue = JobQueue()
    jobqueue.push(SampleJob.new(issued_by="worker_1"))
    jobqueue.push(EvaluateJob.new(issued_by="worker_1", trial_id="1"))

    def _mutate(jobqueue: JobQueue) -> None:
        jobqueue.pop()
        jobqueue.pop()

    return jobqueue.as_filesystem_shared(tmp_path / "jobqueue"), _mutate

@case
def case_seed_snapshot(tmp_path: Path) -> tuple[Shared[SeedSnapshot, Path], Callable[[SeedSnapshot], None]]:
    seed = SeedSnapshot.new_capture()
    def _mutate(seed: SeedSnapshot) -> None:
        random.randint(0, 100)
        seed.capture()

    return seed.as_filesystem_shared(tmp_path / "seed_state"), _mutate

@parametrize_with_cases("shared, mutate", cases=".")
def test_initial_state(shared: Shared, mutate: Callable) -> None:
    assert shared.version() is not None
    assert shared.is_locked() == False
    assert shared.is_stale() == False
    assert shared.current() == shared.pull_latest()


@parametrize_with_cases("shared, mutate", cases=".")
def test_commit_updates_version_and_latest(shared: Shared, mutate: Callable) -> None:
    current_data = shared.current()
    current_version = shared.version()

    new_data = copy.deepcopy(current_data)
    mutate(new_data)
    assert new_data != current_data

    new_version = shared.commit(new_data)
    assert new_version != current_version
    assert shared.version() == new_version
    assert shared.current() == new_data
    assert shared.pull_latest() == new_data
    assert shared.is_stale() == False
    assert shared.is_locked() == False

@parametrize_with_cases("shared1, mutate", cases=".")
def test_share_acquire_mutate_and_commit(shared1: Shared, mutate: Callable) -> None:
    shared2 = shared1.deepcopy()
    assert shared1 == shared2
    assert not shared1.is_locked()
    assert not shared2.is_locked()

    with shared2.acquire() as (data2, commit):
        assert shared1.is_locked()
        assert shared2.is_locked()
        mutate(data2)
        commit(data2)
        assert shared1.version() != shared2.version()

    assert not shared1.is_locked()
    assert not shared2.is_locked()

    assert shared1 != shared2
    assert shared1.version() != shared2.version()
    assert shared1.current() != shared2.current()
    assert shared1.is_stale()

    shared1.pull_latest()
    assert not shared1.is_stale()
    assert not shared2.is_stale()
    assert shared1.current() == shared2.current()


@parametrize_with_cases("shared1, mutate", cases=".")
def test_share_unsafe_pull_latest(shared1: Shared, mutate: Callable) -> None:
    shared2 = shared1.deepcopy()
    assert shared1 == shared2
    assert not shared1.is_locked()
    assert not shared2.is_locked()

    with shared2.acquire() as (data2, commit):
        mutate(data2)
        commit(data2)
        assert shared1.version() != shared2.version()
        shared1.unsafe_pull_latest()
        assert shared1.version() == shared2.version()
        assert shared1.current() == shared2.current()

@parametrize_with_cases("shared, mutate", cases=".")
def test_shared_new_fails_if_done_on_existing_resource(shared: Shared, mutate: Callable) -> None:
    data, version, store, lock = shared.components()
    with pytest.raises(Shared.ResourceExistsError):
        shared.new(data, store=store, locker=lock)
