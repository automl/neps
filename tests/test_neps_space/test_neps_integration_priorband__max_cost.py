from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import neps
from neps import algorithms
from neps.space.neps_spaces.parameters import (
    ConfidenceLevel,
    Fidelity,
    Float,
    Integer,
    Pipeline,
)

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


class DemoHyperparameterWithFidelitySpace(Pipeline):
    float1 = Float(
        min_value=1,
        max_value=1000,
        log=False,
        prior=600,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    float2 = Float(
        min_value=-100,
        max_value=100,
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer1 = Integer(
        min_value=0,
        max_value=500,
        prior=35,
        prior_confidence=ConfidenceLevel.LOW,
    )
    fidelity = Fidelity(
        domain=Integer(
            min_value=1,
            max_value=100,
        ),
    )


@pytest.mark.parametrize(
    ("optimizer", "optimizer_name"),
    [
        (
            algorithms.neps_random_search,
            "new__RandomSearch",
        ),
        (
            algorithms.neps_complex_random_search,
            "new__ComplexRandomSearch",
        ),
        (
            partial(algorithms.neps_priorband, base="successive_halving"),
            "new__priorband+successive_halving",
        ),
        (
            partial(algorithms.neps_priorband, base="asha"),
            "new__priorband+asha",
        ),
        (
            partial(algorithms.neps_priorband, base="async_hb"),
            "new__priorband+async_hb",
        ),
        (
            algorithms.neps_priorband,
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
        evaluate_pipeline=evaluate_pipeline,
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
            partial(algorithms.priorband, base="successive_halving"),
            "old__priorband+successive_halving",
        ),
        (
            partial(algorithms.priorband, base="asha"),
            "old__priorband+asha",
        ),
        (
            partial(algorithms.priorband, base="async_hb"),
            "old__priorband+async_hb",
        ),
        (
            algorithms.priorband,
            "old__priorband+hyperband",
        ),
    ],
)
def test_hyperparameter_with_fidelity_demo_old(optimizer, optimizer_name):
    optimizer.__name__ = optimizer_name  # Needed by NEPS later.
    pipeline_space = DemoHyperparameterWithFidelitySpace()
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
