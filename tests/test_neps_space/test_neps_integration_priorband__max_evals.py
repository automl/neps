from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import neps
import neps.optimizers.algorithms as old_algorithms
from neps.optimizers import neps_algorithms
from neps.space.neps_spaces.parameters import (
    ConfidenceLevel,
    Fidelity,
    Float,
    Integer,
    Pipeline,
)


def evaluate_pipeline(float1, float2, integer1, fidelity):
    return -float(np.sum([float1, float2, integer1])) * fidelity


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
            neps_algorithms.neps_random_search,
            "new__RandomSearch",
        ),
        (
            neps_algorithms.neps_complex_random_search,
            "new__ComplexRandomSearch",
        ),
        (
            partial(neps_algorithms.neps_priorband, base="successive_halving"),
            "new__priorband+successive_halving",
        ),
        (
            partial(neps_algorithms.neps_priorband, base="asha"),
            "new__priorband+asha",
        ),
        (
            partial(neps_algorithms.neps_priorband, base="async_hb"),
            "new__priorband+async_hb",
        ),
        (
            neps_algorithms.neps_priorband,
            "new__priorband+hyperband",
        ),
    ],
)
def test_hyperparameter_with_fidelity_demo_new(optimizer, optimizer_name):
    optimizer.__name__ = optimizer_name  # Needed by NEPS later.
    pipeline_space = DemoHyperparameterWithFidelitySpace()
    root_directory = f"results/hyperparameter_with_fidelity__evals__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        post_run_summary=True,
        max_evaluations_total=200,
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
    root_directory = f"results/hyperparameter_with_fidelity__evals__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        post_run_summary=True,
        max_evaluations_total=200,
        overwrite_working_directory=True,
    )
    neps.status(root_directory, print_summary=True)
