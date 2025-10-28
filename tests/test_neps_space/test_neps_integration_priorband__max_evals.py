from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import neps
from neps.optimizers import algorithms
from neps.space.neps_spaces.parameters import (
    ConfidenceLevel,
    Fidelity,
    Float,
    Integer,
    PipelineSpace,
)


def evaluate_pipeline(float1, float2, integer1, fidelity):
    return -float(np.sum([float1, float2, integer1])) * fidelity


class DemoHyperparameterWithFidelitySpace(PipelineSpace):
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
            partial(algorithms.neps_random_search, ignore_fidelity=True),
            "new__RandomSearch",
        ),
        (
            partial(algorithms.complex_random_search, ignore_fidelity=True),
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
    root_directory = f"tests_tmpdir/test_neps_spaces/results/hyperparameter_with_fidelity__evals__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        evaluations_to_spend=100,
        overwrite_root_directory=True,
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
    root_directory = f"tests_tmpdir/test_neps_spaces/results/hyperparameter_with_fidelity__evals__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        evaluations_to_spend=100,
        overwrite_root_directory=True,
    )
    neps.status(root_directory, print_summary=True)
