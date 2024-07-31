"""NOTE: These tests are pretty specific to the filebased state implementation.
This could be generalized if we end up with a server based implementation but
for now we're just testing the filebased implementation."""

from pathlib import Path
from typing import Any

import pytest
from neps.optimizers.base_optimizer import BaseOptimizer
from neps.search_spaces.hyperparameters import (
    FloatParameter,
    IntegerParameter,
    ConstantParameter,
    CategoricalParameter,
)
from neps.search_spaces.search_space import SearchSpace
from neps.state.filebased import (
    create_or_load_filebased_neps_state,
)

from pytest_cases import fixture, parametrize, parametrize_with_cases, case
from neps.state.neps_state import NePSState
from neps.state.optimizer import BudgetInfo, OptimizationState, OptimizerInfo
from neps.optimizers import SearcherMapping
from neps.utils.common import MissingDependencyError


@case
def case_search_space_no_fid() -> SearchSpace:
    return SearchSpace(
        a=FloatParameter(0, 1),
        b=CategoricalParameter(["a", "b", "c"]),
        c=ConstantParameter("a"),
        d=IntegerParameter(0, 10),
    )


@case
def case_search_space_with_fid() -> SearchSpace:
    return SearchSpace(
        a=FloatParameter(0, 1),
        b=CategoricalParameter(["a", "b", "c"]),
        c=ConstantParameter("a"),
        d=IntegerParameter(0, 10),
        e=IntegerParameter(1, 10, is_fidelity=True),
    )


@case
def case_search_space_no_fid_with_prior() -> SearchSpace:
    return SearchSpace(
        a=FloatParameter(0, 1, default=0.5),
        b=CategoricalParameter(["a", "b", "c"], default="a"),
        c=ConstantParameter("a"),
        d=IntegerParameter(0, 10, default=5),
    )


@case
def case_search_space_fid_with_prior() -> SearchSpace:
    return SearchSpace(
        a=FloatParameter(0, 1, default=0.5),
        b=CategoricalParameter(["a", "b", "c"], default="a"),
        c=ConstantParameter("a"),
        d=IntegerParameter(0, 10, default=5),
        e=IntegerParameter(1, 10, is_fidelity=True),
    )


# See issue #118
NON_INSTANTIABLE_SEARCH_SPACES_WITHOUT_SPECIFIC_KWARGS = "assisted_regularized_evolution"

# See issue #121
JUST_SKIP = [
    "multifidelity_tpe",
]

#
OPTIMIZER_FAILS_WITH_FIDELITY = [
    "random_search",
]

# There's no programattic way to check if a class requires a fidelity.
# See issue #118, #119, #120
OPTIMIZER_REQUIRES_FIDELITY = [
    "successive_halving",
    "successive_halving_prior",
    "asha",
    "asha_prior",
    "hyperband",
    "hyperband_custom_default",
    "priorband",
    "mobster",
    "mf_ei_bo",
]
OPTIMIZER_REQUIRES_BUDGET = [
    "successive_halving_prior",
    "hyperband_custom_default",
    "asha",
    "priorband",
    "hyperband",
    "asha_prior",
    "mobster",
]
REQUIRES_PRIOR = {
    "priorband",
}
REQUIRES_COST = ["cost_cooling_bayesian_optimization", "cost_cooling"]


@fixture
@parametrize(
    "key",
    [
        k
        for k in SearcherMapping.keys()
        if k not in NON_INSTANTIABLE_SEARCH_SPACES_WITHOUT_SPECIFIC_KWARGS
    ],
)
@parametrize_with_cases("search_space", cases=".", prefix="case_search_space")
def optimizer_and_key(key: str, search_space: SearchSpace) -> tuple[BaseOptimizer, str]:
    if key in JUST_SKIP:
        pytest.xfail(f"{key} is not instantiable")

    if key in REQUIRES_PRIOR and search_space.hyperparameters["a"].default is None:
        pytest.xfail(f"{key} requires a prior")

    if search_space.has_fidelity and key in OPTIMIZER_FAILS_WITH_FIDELITY:
        pytest.xfail(f"{key} crashed with a fidelity")

    if key in OPTIMIZER_REQUIRES_FIDELITY and not search_space.has_fidelity:
        pytest.xfail(f"{key} requires a fidelity parameter")
    kwargs: dict[str, Any] = {
        "pipeline_space": search_space,
    }
    if key in OPTIMIZER_REQUIRES_BUDGET:
        kwargs["budget"] = 10

    optimizer_cls = SearcherMapping[key]

    try:
        return optimizer_cls(**kwargs), key
    except MissingDependencyError as e:
        pytest.xfail(f"{key} requires {e.dep} to run.")


@parametrize("optimizer_info", [OptimizerInfo({"a": "b"}), OptimizerInfo({})])
@parametrize("budget", [BudgetInfo(max_cost_budget=10, used_cost_budget=0), None])
@parametrize("shared_state", [{"a": "b"}, {}])
def case_neps_state_filebased(
    tmp_path: Path,
    budget: BudgetInfo | None,
    optimizer_info: OptimizerInfo,
    shared_state: dict[str, Any],
) -> NePSState:
    new_path = tmp_path / "neps_state"
    return create_or_load_filebased_neps_state(
        directory=new_path,
        optimizer_info=optimizer_info,
        optimizer_state=OptimizationState(budget=budget, shared_state=shared_state),
    )


@parametrize_with_cases("neps_state", cases=".", prefix="case_neps_state")
def test_sample_trial(
    neps_state: NePSState,
    optimizer_and_key: tuple[BaseOptimizer, str],
) -> None:
    optimizer, key = optimizer_and_key
    if key in REQUIRES_COST and neps_state.optimizer_state().budget is None:
        pytest.xfail(f"{key} requires a cost budget")

    assert neps_state.get_all_trials() == {}
    assert neps_state.get_next_pending_trial() is None
    assert neps_state.get_next_pending_trial(n=10) == []
    assert neps_state.all_trial_ids() == set()

    trial1 = neps_state.sample_trial(optimizer=optimizer, worker_id="1")
    for k, v in trial1.config.items():
        assert k in optimizer.pipeline_space.hyperparameters
        assert v is not None, f"'{k}' is None in {trial1.config}"

    assert neps_state.get_all_trials() == {trial1.id: trial1}
    assert neps_state.get_next_pending_trial() == trial1
    assert neps_state.get_next_pending_trial(n=10) == [trial1]
    assert neps_state.all_trial_ids() == {trial1.id}

    trial2 = neps_state.sample_trial(optimizer=optimizer, worker_id="1")
    for k, v in trial1.config.items():
        assert k in optimizer.pipeline_space.hyperparameters
        assert v is not None, f"'{k}' is None in {trial1.config}"

    assert trial1 != trial2

    assert neps_state.get_all_trials() == {trial1.id: trial1, trial2.id: trial2}
    assert neps_state.get_next_pending_trial() == trial1
    assert neps_state.get_next_pending_trial(n=10) == [trial1, trial2]
    assert neps_state.all_trial_ids() == {trial1.id, trial2.id}
