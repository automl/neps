from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import neps
import neps.optimizers.algorithms as old_algorithms
import neps.space.neps_spaces.optimizers.algorithms
import neps.space.neps_spaces.optimizers.bracket_optimizer as new_bracket_optimizer
import neps.space.neps_spaces.parameters
from neps.space.neps_spaces import neps_space

_COSTS = {}


def evaluate_pipeline(float1, float2, integer1, fidelity):
    objective_to_minimize = -float(np.sum([float1, float2, integer1])) * fidelity

    key = (float1, float2, integer1)
    old_cost = _COSTS.get(key, 0)
    added_cost = fidelity - old_cost

    _COSTS[key] = fidelity

    return {
        "objective_to_minimize": objective_to_minimize,
        "cost": added_cost,
    }


old_pipeline_space = {
    "float1": neps.Float(
        lower=1,
        upper=1000,
        log=False,
        prior=600,
        prior_confidence="medium",
    ),
    "float2": neps.Float(
        lower=-100,
        upper=100,
        prior=0,
        prior_confidence="medium",
    ),
    "integer1": neps.Integer(
        lower=0,
        upper=500,
        prior=35,
        prior_confidence="low",
    ),
    "fidelity": neps.Integer(
        lower=1,
        upper=100,
        is_fidelity=True,
    ),
}


class DemoHyperparameterWithFidelitySpace(neps.space.neps_spaces.parameters.Pipeline):
    float1 = neps.space.neps_spaces.parameters.Float(
        min_value=1,
        max_value=1000,
        log=False,
        prior=600,
        prior_confidence=neps.space.neps_spaces.parameters.ConfidenceLevel.MEDIUM,
    )
    float2 = neps.space.neps_spaces.parameters.Float(
        min_value=-100,
        max_value=100,
        prior=0,
        prior_confidence=neps.space.neps_spaces.parameters.ConfidenceLevel.MEDIUM,
    )
    integer1 = neps.space.neps_spaces.parameters.Integer(
        min_value=0,
        max_value=500,
        prior=35,
        prior_confidence=neps.space.neps_spaces.parameters.ConfidenceLevel.LOW,
    )
    fidelity = neps.space.neps_spaces.parameters.Fidelity(
        domain=neps.space.neps_spaces.parameters.Integer(
            min_value=1,
            max_value=100,
        ),
    )


@pytest.mark.parametrize(
    ("optimizer", "optimizer_name"),
    [
        (
            neps.space.neps_spaces.optimizers.algorithms.RandomSearch,
            "new__RandomSearch",
        ),
        (
            neps.space.neps_spaces.optimizers.algorithms.ComplexRandomSearch,
            "new__ComplexRandomSearch",
        ),
        (
            partial(new_bracket_optimizer.priorband, base="successive_halving"),
            "new__priorband+successive_halving",
        ),
        (
            partial(new_bracket_optimizer.priorband, base="asha"),
            "new__priorband+asha",
        ),
        (
            partial(new_bracket_optimizer.priorband, base="async_hb"),
            "new__priorband+async_hb",
        ),
        (
            new_bracket_optimizer.priorband,
            "new__priorband+hyperband",
        ),
    ],
)
def test_hyperparameter_with_fidelity_demo_new(optimizer, optimizer_name):
    optimizer.__name__ = optimizer_name  # Needed by NEPS later.
    pipeline_space = DemoHyperparameterWithFidelitySpace()
    root_directory = f"results/hyperparameter_with_fidelity__costs__{optimizer.__name__}"

    # Reset the _COSTS global, so they do not get mixed up between tests.
    _COSTS.clear()

    neps.run(
        evaluate_pipeline=neps_space.adjust_evaluation_pipeline_for_neps_space(
            evaluate_pipeline,
            pipeline_space,
        ),
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        post_run_summary=True,
        max_cost_total=1000,
        overwrite_working_directory=True,
    )
    neps.status(root_directory, print_summary=True)


@pytest.mark.parametrize(
    ("optimizer", "optimizer_name"),
    [
        (
            partial(old_algorithms.priorband, base="successive_halving"),
            "old__priorband+successive_halving",
        ),
        (
            partial(old_algorithms.priorband, base="asha"),
            "old__priorband+asha",
        ),
        (
            partial(old_algorithms.priorband, base="async_hb"),
            "old__priorband+async_hb",
        ),
        (
            old_algorithms.priorband,
            "old__priorband+hyperband",
        ),
    ],
)
def test_hyperparameter_with_fidelity_demo_old(optimizer, optimizer_name):
    optimizer.__name__ = optimizer_name  # Needed by NEPS later.
    pipeline_space = old_pipeline_space
    root_directory = f"results/hyperparameter_with_fidelity__costs__{optimizer.__name__}"

    # Reset the _COSTS global, so they do not get mixed up between tests.
    _COSTS.clear()

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        post_run_summary=True,
        max_cost_total=1000,
        overwrite_working_directory=True,
    )
    neps.status(root_directory, print_summary=True)
