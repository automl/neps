"""NOTE: These tests are pretty specific to the filebased state implementation.
This could be generalized if we end up with a server based implementation but
for now we're just testing the filebased implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest_cases import fixture, parametrize

from neps.exceptions import NePSError, TrialNotFoundError
from neps.optimizers import OptimizerInfo
from neps.state.err_dump import ErrDump
from neps.state.neps_state import NePSState
from neps.state.optimizer import BudgetInfo, OptimizationState
from neps.state.seed_snapshot import SeedSnapshot


@fixture
@parametrize("budget_info", [BudgetInfo(max_cost_total=10, used_cost_budget=0), None])
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


def test_create_with_new_filebased_neps_state(
    tmp_path: Path,
    optimizer_info: OptimizerInfo,
    optimizer_state: OptimizationState,
) -> None:
    new_path = tmp_path / "neps_state"
    neps_state = NePSState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=optimizer_state,
    )
    assert neps_state.lock_and_get_optimizer_info() == optimizer_info
    assert neps_state.lock_and_get_optimizer_state() == optimizer_state
    assert neps_state.all_trial_ids() == []
    assert neps_state.lock_and_read_trials() == {}
    assert neps_state.lock_and_get_errors() == ErrDump(errs=[])
    assert neps_state.lock_and_get_next_pending_trial() is None
    assert neps_state.lock_and_get_next_pending_trial(n=10) == []

    with pytest.raises(TrialNotFoundError):
        assert neps_state.lock_and_get_trial_by_id("1")


def test_create_or_load_with_load_filebased_neps_state(
    tmp_path: Path,
    optimizer_info: OptimizerInfo,
    optimizer_state: OptimizationState,
) -> None:
    new_path = tmp_path / "neps_state"
    neps_state = NePSState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=optimizer_state,
    )

    # NOTE: This isn't a defined way to do this but we should check
    # that we prioritize what's in the existing data over what
    # was passed in.
    different_state = OptimizationState(
        budget=BudgetInfo(max_cost_total=20, used_cost_budget=10),
        seed_snapshot=SeedSnapshot.new_capture(),
        shared_state=None,
    )
    neps_state2 = NePSState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=different_state,
    )
    assert neps_state == neps_state2


def test_load_on_existing_neps_state(
    tmp_path: Path,
    optimizer_info: OptimizerInfo,
    optimizer_state: OptimizationState,
) -> None:
    new_path = tmp_path / "neps_state"
    neps_state = NePSState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=optimizer_state,
    )

    neps_state2 = NePSState.create_or_load(path=new_path, load_only=True)
    assert neps_state == neps_state2


def test_new_or_load_on_existing_neps_state_with_different_optimizer_info(
    tmp_path: Path,
    optimizer_info: OptimizerInfo,
    optimizer_state: OptimizationState,
) -> None:
    new_path = tmp_path / "neps_state"
    NePSState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=optimizer_state,
    )

    with pytest.raises(NePSError):
        NePSState.create_or_load(
            path=new_path,
            optimizer_info=OptimizerInfo(name="randomlll", info={"e": "f"}),
            optimizer_state=optimizer_state,
        )
