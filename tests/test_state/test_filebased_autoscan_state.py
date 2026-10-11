"""NOTE: These tests are pretty specific to the filebased state implementation.
This could be generalized if we end up with a server based implementation but
for now we're just testing the filebased implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest_cases import fixture, parametrize

from autoscan.exceptions import AutoScAnError, TrialNotFoundError
from autoscan.optimizers import OptimizerInfo
from autoscan.space.autoscan_spaces.parameters import Float, PipelineSpace
from autoscan.state.autoscan_state import AutoScAnState
from autoscan.state.err_dump import ErrDump
from autoscan.state.optimizer import BudgetInfo, OptimizationState
from autoscan.state.seed_snapshot import SeedSnapshot


@fixture
@parametrize(
    "budget_info",
    [BudgetInfo(worker_cost_to_spend=10, used_cost_budget=0), None],
)
@parametrize("shared_state", [{"a": "b"}, {}])
def optimizer_state(
    budget_info: BudgetInfo | None,
    shared_state: dict[str, Any],
) -> OptimizationState:
    return OptimizationState(
        budget=budget_info,
        seed_snapshot=SeedSnapshot.new_capture(),
        shared_state=shared_state,
    )


@fixture
@parametrize(
    "optimizer_info",
    [OptimizerInfo(name="blah", info={"a": "b"})],
)
def optimizer_info(optimizer_info: OptimizerInfo) -> OptimizerInfo:
    return optimizer_info


def test_create_with_new_filebased_autoscan_state(
    tmp_path: Path,
    optimizer_info: OptimizerInfo,
    optimizer_state: OptimizationState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    new_path = tmp_path / "autoscan_state"
    autoscan_state = AutoScAnState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=optimizer_state,
        pipeline_space=TestSpace(),
    )
    assert autoscan_state.lock_and_get_optimizer_info() == optimizer_info
    assert autoscan_state.lock_and_get_optimizer_state() == optimizer_state
    assert autoscan_state.all_trial_ids() == []
    assert autoscan_state.lock_and_read_trials() == {}
    assert autoscan_state.lock_and_get_errors() == ErrDump(errs=[])
    assert autoscan_state.lock_and_get_next_pending_trial() is None
    assert autoscan_state.lock_and_get_next_pending_trial(n=10) == []

    with pytest.raises(TrialNotFoundError):
        assert autoscan_state.lock_and_get_trial_by_id("1")


def test_create_or_load_with_load_filebased_autoscan_state(
    tmp_path: Path,
    optimizer_info: OptimizerInfo,
    optimizer_state: OptimizationState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    new_path = tmp_path / "autoscan_state"
    autoscan_state = AutoScAnState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=optimizer_state,
        pipeline_space=TestSpace(),
    )

    # NOTE: This isn't a defined way to do this but we should check
    # that we prioritize what's in the existing data over what
    # was passed in.
    different_state = OptimizationState(
        budget=BudgetInfo(worker_cost_to_spend=20, used_cost_budget=10),
        seed_snapshot=SeedSnapshot.new_capture(),
        shared_state=None,
    )
    autoscan_state2 = AutoScAnState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=different_state,
        pipeline_space=TestSpace(),
    )
    assert autoscan_state == autoscan_state2


def test_load_on_existing_autoscan_state(
    tmp_path: Path,
    optimizer_info: OptimizerInfo,
    optimizer_state: OptimizationState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    new_path = tmp_path / "autoscan_state"
    autoscan_state = AutoScAnState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=optimizer_state,
        pipeline_space=TestSpace(),
    )

    autoscan_state2 = AutoScAnState.create_or_load(path=new_path, load_only=True)
    assert autoscan_state == autoscan_state2


def test_pipeline_space_written_and_reloaded(tmp_path: Path) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    optimizer_info = OptimizerInfo(name="test", info={"a": "b"})
    optimizer_state = OptimizationState(
        budget=BudgetInfo(worker_cost_to_spend=10, used_cost_budget=0),
        seed_snapshot=SeedSnapshot.new_capture(),
        shared_state={},
    )

    new_path = tmp_path / "autoscan_state"
    autoscan_state = AutoScAnState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=optimizer_state,
        pipeline_space=TestSpace(),
    )

    # Load-only should return the same state
    autoscan_state2 = AutoScAnState.create_or_load(path=new_path, load_only=True)
    assert autoscan_state == autoscan_state2

    # And their pipeline spaces must serialize to the same bytes
    import pickle

    assert pickle.dumps(autoscan_state._pipeline_space) == pickle.dumps(
        autoscan_state2._pipeline_space
    )


def test_new_or_load_on_existing_autoscan_state_with_different_optimizer_info(
    tmp_path: Path,
    optimizer_info: OptimizerInfo,
    optimizer_state: OptimizationState,
) -> None:
    class TestSpace(PipelineSpace):
        a = Float(0, 1)

    new_path = tmp_path / "autoscan_state"
    AutoScAnState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=optimizer_state,
        pipeline_space=TestSpace(),
    )

    with pytest.raises(AutoScAnError):
        AutoScAnState.create_or_load(
            path=new_path,
            optimizer_info=OptimizerInfo(name="randomlll", info={"e": "f"}),
            optimizer_state=optimizer_state,
            pipeline_space=TestSpace(),
        )
