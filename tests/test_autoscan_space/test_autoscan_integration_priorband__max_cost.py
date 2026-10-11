from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import autoscan
from autoscan import algorithms
from autoscan.space.autoscan_spaces.parameters import (
    ConfidenceLevel,
    Float,
    Integer,
    IntegerFidelity,
    PipelineSpace,
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
    fidelity = IntegerFidelity(
        lower=1,
        upper=100,
    )


@pytest.mark.parametrize(
    ("optimizer", "optimizer_name"),
    [
        (
            partial(algorithms.autoscan_random_search, ignore_fidelity=True),
            "autoscan_random_search",
        ),
        (
            partial(algorithms.complex_random_search, ignore_fidelity=True),
            "autoscan_complex_random_search",
        ),
        (
            partial(algorithms.autoscan_priorband, base="successive_halving"),
            "autoscan_priorband+successive_halving",
        ),
        (
            partial(algorithms.autoscan_priorband, base="asha"),
            "autoscan_priorband+asha",
        ),
        (
            partial(algorithms.autoscan_priorband, base="async_hb"),
            "autoscan_priorband+async_hb",
        ),
        (
            algorithms.autoscan_priorband,
            "autoscan_priorband+hyperband",
        ),
    ],
)
def test_hyperparameter_with_fidelity_demo_new(optimizer, optimizer_name, tmp_path):
    optimizer.__name__ = (
        "autoscan_priorband" if "priorband" in optimizer_name else optimizer_name
    )  # Needed by AutoScAn later.
    pipeline_space = DemoHyperparameterWithFidelitySpace()
    root_directory = tmp_path / f"hyperparameter_with_fidelity__costs__{optimizer_name}"

    # Reset the _COSTS global, so they do not get mixed up between tests.
    _COSTS.clear()

    autoscan.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        worker_cost_to_spend=100,
        overwrite_root_directory=True,
    )
    autoscan.status(root_directory, print_summary=True)


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
def test_hyperparameter_with_fidelity_demo_old(optimizer, optimizer_name, tmp_path):
    optimizer.__name__ = "priorband"  # Needed by AutoScAn later.
    pipeline_space = DemoHyperparameterWithFidelitySpace()
    root_directory = tmp_path / f"hyperparameter_with_fidelity__costs__{optimizer_name}"

    # Reset the _COSTS global, so they do not get mixed up between tests.
    _COSTS.clear()

    autoscan.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        worker_cost_to_spend=100,
        overwrite_root_directory=True,
    )
    autoscan.status(root_directory, print_summary=True)
