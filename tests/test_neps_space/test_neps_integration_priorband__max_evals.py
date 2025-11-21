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
        lower=1,
        upper=1000,
        log=False,
        prior=600,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    float2 = Float(
        lower=-100,
        upper=100,
        prior=0,
        prior_confidence=ConfidenceLevel.MEDIUM,
    )
    integer1 = Integer(
        lower=0,
        upper=500,
        prior=35,
        prior_confidence=ConfidenceLevel.LOW,
    )
    fidelity = Fidelity(
        domain=Integer(
            lower=1,
            upper=100,
        ),
    )


@pytest.mark.parametrize(
    ("optimizer", "optimizer_name"),
    [
        (
            partial(algorithms.neps_random_search, ignore_fidelity=True),
            "neps_random_search",
        ),
        (
            partial(algorithms.complex_random_search, ignore_fidelity=True),
            "neps_complex_random_search",
        ),
        (
            partial(algorithms.neps_priorband, base="successive_halving"),
            "neps_priorband+successive_halving",
        ),
        (
            partial(algorithms.neps_priorband, base="asha"),
            "neps_priorband+asha",
        ),
        (
            partial(algorithms.neps_priorband, base="async_hb"),
            "neps_priorband+async_hb",
        ),
        (
            algorithms.neps_priorband,
            "neps_priorband+hyperband",
        ),
    ],
)
def test_hyperparameter_with_fidelity_demo_new(optimizer, optimizer_name):
    optimizer.__name__ = (
        "neps_priorband" if "priorband" in optimizer_name else optimizer_name
    )  # Needed by NEPS later.
    pipeline_space = DemoHyperparameterWithFidelitySpace()
    root_directory = f"tests_tmpdir/test_neps_spaces/results/hyperparameter_with_fidelity__evals__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        fidelities_to_spend=50 if "priorband" in optimizer.__name__ else None,
        evaluations_to_spend=50 if "priorband" not in optimizer.__name__ else None,
        overwrite_root_directory=True,
    )
    neps.status(root_directory, print_summary=True)


@pytest.mark.parametrize(
    ("optimizer", "optimizer_name"),
    [
        (
            partial(algorithms.priorband, base="successive_halving"),
            "old_priorband+successive_halving",
        ),
        (
            partial(algorithms.priorband, base="asha"),
            "old_priorband+asha",
        ),
        (
            partial(algorithms.priorband, base="async_hb"),
            "old_priorband+async_hb",
        ),
        (
            algorithms.priorband,
            "old_priorband+hyperband",
        ),
    ],
)
def test_hyperparameter_with_fidelity_demo_old(optimizer, optimizer_name):
    optimizer.__name__ = "priorband"  # Needed by NEPS later.
    pipeline_space = DemoHyperparameterWithFidelitySpace()
    root_directory = f"tests_tmpdir/test_neps_spaces/results/hyperparameter_with_fidelity__evals__{optimizer.__name__}"

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        fidelities_to_spend=50,
        overwrite_root_directory=True,
    )
    neps.status(root_directory, print_summary=True)
