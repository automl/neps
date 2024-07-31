from __future__ import annotations

from pytest_cases import parametrize, parametrize_with_cases, case
import copy
import numpy as np
import random
from neps.state.err_dump import ErrDump, SerializableTrialError
from neps.state.filebased import (
    ReaderWriterErrDump,
    ReaderWriterOptimizationState,
    ReaderWriterOptimizerInfo,
    ReaderWriterSeedSnapshot,
    ReaderWriterTrial,
    FileVersioner,
    FileLocker,
)
from neps.state.optimizer import BudgetInfo, OptimizationState, OptimizerInfo
from neps.state.protocols import Synced
from neps.state.trial import Trial
import pytest
from typing import Any, Callable
from pathlib import Path
from neps.state import SeedSnapshot, Synced, Trial


@case
def case_trial_1(tmp_path: Path) -> tuple[Synced[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        location="",
        config={"a": "b"},
        time_sampled=0,
        previous_trial=None,
        previous_trial_location=None,
        worker_id=0,
    )

    def _mutate(trial: Trial) -> None:
        trial.set_submitted(time_submitted=1)

    x = Synced.new(
        data=trial,
        location=tmp_path / "1",
        locker=FileLocker(lock_path=tmp_path / "1" / ".lock", poll=0.1, timeout=None),
        versioner=FileVersioner(version_file=tmp_path / "1" / ".version"),
        reader_writer=ReaderWriterTrial(),
    )
    return x, _mutate


@case
def case_trial_2(tmp_path: Path) -> tuple[Synced[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        location="",
        config={"a": "b"},
        time_sampled=0,
        previous_trial=None,
        previous_trial_location=None,
        worker_id=0,
    )
    trial.set_submitted(time_submitted=1)

    def _mutate(trial: Trial) -> None:
        trial.set_evaluating(time_started=2, worker_id="1")

    x = Synced.new(
        data=trial,
        location=tmp_path / "1",
        locker=FileLocker(lock_path=tmp_path / "1" / ".lock", poll=0.1, timeout=None),
        versioner=FileVersioner(version_file=tmp_path / "1" / ".version"),
        reader_writer=ReaderWriterTrial(),
    )
    return x, _mutate


@case
def case_trial_3(tmp_path: Path) -> tuple[Synced[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        location="",
        time_sampled=0,
        previous_trial=None,
        previous_trial_location=None,
        worker_id=0,
    )
    trial.set_submitted(time_submitted=1)
    trial.set_evaluating(time_started=2, worker_id="1")

    def _mutate(trial: Trial) -> None:
        trial.set_complete(
            time_end=3,
            loss=1,
            cost=1,
            extra={"hi": [1, 2, 3]},
            learning_curve=[1],
            report_as="success",
            evaluation_duration=1,
            err=None,
            tb=None,
        )

    x = Synced.new(
        data=trial,
        location=tmp_path / "1",
        locker=FileLocker(lock_path=tmp_path / "1" / ".lock", poll=0.1, timeout=None),
        versioner=FileVersioner(version_file=tmp_path / "1" / ".version"),
        reader_writer=ReaderWriterTrial(),
    )
    return x, _mutate


@case
def case_trial_4(tmp_path: Path) -> tuple[Synced[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        location="",
        time_sampled=0,
        previous_trial=None,
        previous_trial_location=None,
        worker_id=0,
    )
    trial.set_submitted(time_submitted=1)
    trial.set_evaluating(time_started=2, worker_id="1")

    def _mutate(trial: Trial) -> None:
        trial.set_complete(
            time_end=3,
            loss=np.nan,
            cost=np.inf,
            extra={"hi": [1, 2, 3]},
            report_as="failed",
            learning_curve=None,
            evaluation_duration=2,
            err=None,
            tb=None,
        )

    x = Synced.new(
        data=trial,
        location=tmp_path / "1",
        locker=FileLocker(lock_path=tmp_path / "1" / ".lock", poll=0.1, timeout=None),
        versioner=FileVersioner(version_file=tmp_path / "1" / ".version"),
        reader_writer=ReaderWriterTrial(),
    )
    return x, _mutate


@case
def case_trial_5(tmp_path: Path) -> tuple[Synced[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        location="",
        time_sampled=0,
        previous_trial=None,
        previous_trial_location=None,
        worker_id=0,
    )
    trial.set_submitted(time_submitted=1)
    trial.set_evaluating(time_started=2, worker_id=1)

    def _mutate(trial: Trial) -> None:
        trial.set_complete(
            time_end=3,
            loss=np.nan,
            cost=np.inf,
            extra={"hi": [1, 2, 3]},
            learning_curve=None,
            evaluation_duration=2,
            report_as="failed",
            err=ValueError("hi"),
            tb="something something traceback",
        )

    x = Synced.new(
        data=trial,
        location=tmp_path / "1",
        locker=FileLocker(lock_path=tmp_path / "1" / ".lock", poll=0.1, timeout=None),
        versioner=FileVersioner(version_file=tmp_path / "1" / ".version"),
        reader_writer=ReaderWriterTrial(),
    )
    return x, _mutate


@case
def case_trial_6(tmp_path: Path) -> tuple[Synced[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        location="",
        time_sampled=0,
        previous_trial=None,
        previous_trial_location=None,
        worker_id=0,
    )
    trial.set_submitted(time_submitted=1)
    trial.set_evaluating(time_started=2, worker_id=1)

    def _mutate(trial: Trial) -> None:
        trial.set_corrupted()

    x = Synced.new(
        data=trial,
        location=tmp_path / "1",
        locker=FileLocker(lock_path=tmp_path / "1" / ".lock", poll=0.1, timeout=None),
        versioner=FileVersioner(version_file=tmp_path / "1" / ".version"),
        reader_writer=ReaderWriterTrial(),
    )
    return x, _mutate


@case
def case_trial_7(tmp_path: Path) -> tuple[Synced[Trial, Path], Callable[[Trial], None]]:
    trial_id = "1"
    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        location="",
        time_sampled=0,
        previous_trial=None,
        previous_trial_location=None,
        worker_id=0,
    )
    trial.set_submitted(time_submitted=1)
    trial.set_evaluating(time_started=2, worker_id=1)
    trial.set_complete(
        time_end=3,
        loss=np.nan,
        cost=np.inf,
        extra={"hi": [1, 2, 3]},
        learning_curve=[1, 2, 3],
        report_as="failed",
        evaluation_duration=2,
        err=ValueError("hi"),
        tb="something something traceback",
    )

    def _mutate(trial: Trial) -> None:
        trial.reset()

    x = Synced.new(
        data=trial,
        location=tmp_path / "1",
        locker=FileLocker(lock_path=tmp_path / "1" / ".lock", poll=0.1, timeout=None),
        versioner=FileVersioner(version_file=tmp_path / "1" / ".version"),
        reader_writer=ReaderWriterTrial(),
    )
    return x, _mutate


@case
def case_seed_snapshot(
    tmp_path: Path,
) -> tuple[Synced[SeedSnapshot, Path], Callable[[SeedSnapshot], None]]:
    seed = SeedSnapshot.new_capture()

    def _mutate(seed: SeedSnapshot) -> None:
        random.randint(0, 100)
        seed.recapture()

    x = Synced.new(
        data=seed,
        location=tmp_path / "seeds",
        locker=FileLocker(lock_path=tmp_path / "seeds" / ".lock", poll=0.1, timeout=None),
        versioner=FileVersioner(version_file=tmp_path / "seeds" / ".version"),
        reader_writer=ReaderWriterSeedSnapshot(),
    )
    return x, _mutate


@case
@parametrize(
    "err",
    [
        None,
        SerializableTrialError(
            trial_id="1",
            worker_id="2",
            err_type="ValueError",
            err="hi",
            tb="traceback\nmore",
        ),
    ],
)
def case_err_dump(
    tmp_path: Path,
    err: None | SerializableTrialError,
) -> tuple[Synced[ErrDump, Path], Callable[[ErrDump], None]]:
    err_dump = ErrDump() if err is None else ErrDump(errs=[err])

    def _mutate(err_dump: ErrDump) -> None:
        new_err = SerializableTrialError(
            trial_id="2",
            worker_id="2",
            err_type="RuntimeError",
            err="hi",
            tb="traceback\nless",
        )
        err_dump.append(new_err)

    x = Synced.new(
        data=err_dump,
        location=tmp_path / "err_dump",
        locker=FileLocker(
            lock_path=tmp_path / "err_dump" / ".lock", poll=0.1, timeout=None
        ),
        versioner=FileVersioner(version_file=tmp_path / "err_dump" / ".version"),
        reader_writer=ReaderWriterErrDump("all"),
    )
    return x, _mutate


@case
def case_optimizer_info(
    tmp_path: Path,
) -> tuple[Synced[OptimizerInfo, Path], Callable[[OptimizerInfo], None]]:
    optimizer_info = OptimizerInfo(info={"a": "b"})

    def _mutate(optimizer_info: OptimizerInfo) -> None:
        optimizer_info.info["b"] = "c"  # type: ignore # NOTE: We shouldn't be mutating but anywho...

    x = Synced.new(
        data=optimizer_info,
        location=tmp_path / "optimizer_info",
        locker=FileLocker(
            lock_path=tmp_path / "optimizer_info" / ".lock", poll=0.1, timeout=None
        ),
        versioner=FileVersioner(version_file=tmp_path / "optimizer_info" / ".version"),
        reader_writer=ReaderWriterOptimizerInfo(),
    )
    return x, _mutate


@case
@pytest.mark.parametrize(
    "budget", (None, BudgetInfo(max_cost_budget=10, used_cost_budget=0))
)
@pytest.mark.parametrize("shared_state", ({}, {"a": "b"}))
def case_optimization_state(
    tmp_path: Path,
    budget: BudgetInfo | None,
    shared_state: dict[str, Any],
) -> tuple[Synced[OptimizationState, Path], Callable[[OptimizationState], None]]:
    optimization_state = OptimizationState(budget=budget, shared_state=shared_state)

    def _mutate(optimization_state: OptimizationState) -> None:
        optimization_state.shared_state["a"] = "c"  # type: ignore # NOTE: We shouldn't be mutating but anywho...
        optimization_state.budget = BudgetInfo(max_cost_budget=10, used_cost_budget=5)

    x = Synced.new(
        data=optimization_state,
        location=tmp_path / "optimizer_info",
        locker=FileLocker(
            lock_path=tmp_path / "optimizer_info" / ".lock", poll=0.1, timeout=None
        ),
        versioner=FileVersioner(version_file=tmp_path / "optimizer_info" / ".version"),
        reader_writer=ReaderWriterOptimizationState(),
    )
    return x, _mutate


@parametrize_with_cases("shared, mutate", cases=".")
def test_initial_state(shared: Synced, mutate: Callable) -> None:
    assert shared._is_locked() == False
    assert shared._is_stale() == False
    assert shared._unsynced() == shared.synced()


@parametrize_with_cases("shared, mutate", cases=".")
def test_put_updates_current_data_and_is_not_stale(
    shared: Synced, mutate: Callable
) -> None:
    current_data = shared._unsynced()

    new_data = copy.deepcopy(current_data)
    mutate(new_data)
    assert new_data != current_data

    shared.put(new_data)
    assert shared._unsynced() == new_data
    assert shared._is_stale() == False
    assert shared._is_locked() == False


@parametrize_with_cases("shared1, mutate", cases=".")
def test_share_synced_mutate_and_put(shared1: Synced, mutate: Callable) -> None:
    shared2 = shared1.clone()
    assert shared1 == shared2
    assert not shared1._is_locked()
    assert not shared2._is_locked()

    with shared2.acquire() as (data2, put2):
        assert shared1._is_locked()
        assert shared2._is_locked()
        mutate(data2)
        put2(data2)

    assert not shared1._is_locked()
    assert not shared2._is_locked()

    assert shared1 != shared2
    assert shared1._unsynced() != shared2._unsynced()
    assert shared1._is_stale()

    shared1.synced()
    assert not shared1._is_stale()
    assert not shared2._is_stale()
    assert shared1._unsynced() == shared2._unsynced()


@parametrize_with_cases("shared, mutate", cases=".")
def test_shared_new_fails_if_done_on_existing_resource(
    shared: Synced, mutate: Callable
) -> None:
    data, location, versioner, rw, lock = shared._components()
    with pytest.raises(Synced.VersionedResourceAlreadyExistsError):
        Synced.new(
            data=data,
            location=location,
            versioner=versioner,
            reader_writer=rw,
            locker=lock,
        )
