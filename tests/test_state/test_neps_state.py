"""NOTE: These tests are pretty specific to the filebased state implementation.
This could be generalized if we end up with a server based implementation but
for now we're just testing the filebased implementation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from pytest_cases import case, fixture, parametrize, parametrize_with_cases

import neps
from neps.optimizers import (
    AskFunction,
    OptimizerInfo,
    PredefinedOptimizers,
    load_optimizer,
)
from neps.optimizers.ask_and_tell import AskAndTell
from neps.space import SearchSpace
from neps.space.neps_spaces.parameters import (
    Categorical,
    Fidelity,
    Float,
    Integer,
    Pipeline,
)
from neps.state import BudgetInfo, NePSState, OptimizationState, SeedSnapshot


@case
def case_search_space_no_fid() -> Pipeline:
    class Space(Pipeline):
        a = Float(0, 1)
        b = Categorical(("a", "b", "c"))
        c = "a"
        d = Integer(0, 10)

    return Space()


@case
def case_search_space_with_fid() -> Pipeline:
    class SpaceFid(Pipeline):
        a = Float(0, 1)
        b = Categorical(("a", "b", "c"))
        c = "a"
        d = Integer(0, 10)
        e = Fidelity(Integer(1, 10))

    return SpaceFid()


@case
def case_search_space_no_fid_with_prior() -> Pipeline:
    class SpacePrior(Pipeline):
        a = Float(0, 1, prior=0.5)
        b = Categorical(("a", "b", "c"), prior=0)
        c = "a"
        d = Integer(0, 10, prior=5)

    return SpacePrior()


@case
def case_search_space_fid_with_prior() -> Pipeline:
    class SpaceFidPrior(Pipeline):
        a = Float(0, 1, prior=0.5)
        b = Categorical(("a", "b", "c"), prior=0)
        c = "a"
        d = Integer(0, 10, prior=5)
        e = Fidelity(Integer(1, 10))

    return SpaceFidPrior()


# See issue #121
JUST_SKIP = [
    "multifidelity_tpe",
]

# There's no programattic way to check if a class requires or
# doesnt support a fidelity/prior.
# See issue #118, #119, #120
# For now, keep these lists up to date manually and xfail the tests
# that require a fidelity/prior.
REQUIRES_FIDELITY = [
    "successive_halving",
    "asha",
    "hyperband",
    "async_hb",
    "ifbo",
    "priorband",
    "moasha",
    "mo_hyperband",
    "neps_priorband",
]
NO_DEFAULT_FIDELITY_SUPPORT = [
    "random_search",
    "grid_search",
    "bayesian_optimization",
    "pibo",
]
NO_DEFAULT_PRIOR_SUPPORT = [
    "grid_search",
    "bayesian_optimization",
    "ifbo",
    "successive_halving",
    "asha",
    "hyperband",
    "async_hb",
    "random_search",
    "moasha",
    "mo_hyperband",
]
REQUIRES_PRIOR = [
    "pibo",
    "priorband",
]

REQUIRES_NEPS_SPACE = [
    "neps_priorband",
    "neps_random_search",
    "neps_complex_random_search",
]


@fixture
@parametrize("key", list(PredefinedOptimizers.keys()))
@parametrize_with_cases("search_space", cases=".", prefix="case_search_space")
def optimizer_and_key_and_search_space(
    key: str, search_space: Pipeline
) -> tuple[AskFunction, str, Pipeline | SearchSpace]:
    if key in JUST_SKIP:
        pytest.xfail(f"{key} is not instantiable")

    if key in NO_DEFAULT_PRIOR_SUPPORT and any(
        parameter.has_prior if hasattr(parameter, "has_prior") else False
        for parameter in search_space.get_attrs().values()
    ):
        pytest.xfail(f"{key} crashed with a prior")

    if search_space.fidelity_attrs and key in NO_DEFAULT_FIDELITY_SUPPORT:
        pytest.xfail(f"{key} crashed with a fidelity")

    if key in REQUIRES_FIDELITY and not search_space.fidelity_attrs:
        pytest.xfail(f"{key} requires a fidelity parameter")

    if key in REQUIRES_PRIOR and not any(
        parameter.has_prior if hasattr(parameter, "has_prior") else False
        for parameter in search_space.get_attrs().values()
    ):
        pytest.xfail(f"{key} requires a prior")

    kwargs: dict[str, Any] = {}
    opt, _ = load_optimizer((key, kwargs), search_space)  # type: ignore
    converted_space = (
        neps.space.neps_spaces.neps_space.convert_neps_to_classic_search_space(
            search_space
        )
    )
    return (
        opt,
        key,
        converted_space
        if converted_space and key not in REQUIRES_NEPS_SPACE
        else search_space,
    )


@parametrize("optimizer_info", [OptimizerInfo(name="blah", info={"a": "b"})])
@parametrize("max_cost_total", [BudgetInfo(max_cost_total=10, used_cost_budget=0), None])
@parametrize("shared_state", [{"a": "b"}, {}])
def case_neps_state_filebased(
    tmp_path: Path,
    max_cost_total: BudgetInfo | None,
    optimizer_info: OptimizerInfo,
    shared_state: dict[str, Any],
) -> NePSState:
    new_path = tmp_path / "neps_state"
    return NePSState.create_or_load(
        path=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=OptimizationState(
            budget=max_cost_total,
            seed_snapshot=SeedSnapshot.new_capture(),
            shared_state=shared_state,
        ),
    )


@parametrize_with_cases("neps_state", cases=".", prefix="case_neps_state")
def test_sample_trial(
    neps_state: NePSState,
    optimizer_and_key_and_search_space: tuple[AskFunction, str, Pipeline | SearchSpace],
    capsys,
) -> None:
    optimizer, key, search_space = optimizer_and_key_and_search_space

    assert neps_state.lock_and_read_trials() == {}
    assert neps_state.lock_and_get_next_pending_trial() is None
    assert neps_state.lock_and_get_next_pending_trial(n=10) == []
    assert neps_state.all_trial_ids() == []

    trial1 = neps_state.lock_and_sample_trial(optimizer=optimizer, worker_id="1")
    for k, v in trial1.config.items():
        assert v is not None, f"'{k}' is None in {trial1.config}"

    if isinstance(search_space, SearchSpace):
        for name in search_space:
            assert name in trial1.config, f"'{name}' is not in {trial1.config}"
    else:
        config = neps.space.neps_spaces.neps_space.NepsCompatConverter().from_neps_config(
            trial1.config
        )
        resolved_pipeline, _ = neps.space.neps_spaces.neps_space.resolve(
            pipeline=search_space,
            domain_sampler=neps.space.neps_spaces.neps_space.OnlyPredefinedValuesSampler(
                predefined_samplings=config.predefined_samplings
            ),
            environment_values=config.environment_values,
        )
        for name in search_space.get_attrs():
            assert name in resolved_pipeline.get_attrs(), (
                f"'{name}' is not in {resolved_pipeline.get_attrs()}"
            )

    # HACK: Unfortunatly due to windows, who's time.time() is not very
    # precise, we need to introduce a sleep -_-
    time.sleep(0.1)

    assert neps_state.lock_and_read_trials() == {trial1.id: trial1}
    assert neps_state.lock_and_get_next_pending_trial() == trial1
    assert neps_state.lock_and_get_next_pending_trial(n=10) == [trial1]
    assert neps_state.all_trial_ids() == [trial1.id]

    trial2 = neps_state.lock_and_sample_trial(optimizer=optimizer, worker_id="1")
    for k, v in trial1.config.items():
        assert v is not None, f"'{k}' is None in {trial1.config}"

    if isinstance(search_space, SearchSpace):
        for name in search_space:
            assert name in trial1.config, f"'{name}' is not in {trial1.config}"
    else:
        config = neps.space.neps_spaces.neps_space.NepsCompatConverter().from_neps_config(
            trial1.config
        )
        resolved_pipeline, _ = neps.space.neps_spaces.neps_space.resolve(
            pipeline=search_space,
            domain_sampler=neps.space.neps_spaces.neps_space.OnlyPredefinedValuesSampler(
                predefined_samplings=config.predefined_samplings
            ),
            environment_values=config.environment_values,
        )
        for name in search_space.get_attrs():
            assert name in resolved_pipeline.get_attrs(), (
                f"'{name}' is not in {resolved_pipeline.get_attrs()}"
            )

    assert trial1 != trial2

    assert neps_state.lock_and_read_trials() == {trial1.id: trial1, trial2.id: trial2}
    assert neps_state.lock_and_get_next_pending_trial() == trial1
    assert neps_state.lock_and_get_next_pending_trial(n=10) == [trial1, trial2]
    assert sorted(neps_state.all_trial_ids()) == [trial1.id, trial2.id]


def test_optimizers_work_roughly(
    optimizer_and_key_and_search_space: tuple[AskFunction, str, Pipeline | SearchSpace],
) -> None:
    opt, key, search_space = optimizer_and_key_and_search_space
    ask_and_tell = AskAndTell(opt)

    for _ in range(20):
        trial = ask_and_tell.ask()
        ask_and_tell.tell(trial, 1.0)
