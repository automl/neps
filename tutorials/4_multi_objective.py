# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Multi-Objective Optimization with AutoScAn
# This tutorial covers a typical multi-objective optimization workflow with AutoScAn,
# built around **PriMO**, AutoScAn's prior-informed multi-objective optimizer. We run
# it over a synthetic mathematical benchmark with expert priors layered on top. The
# same principles carry over unchanged to real-world pipelines (e.g. minimizing
# both validation loss and model size).

# ## Installation and Setup


# !git clone --depth 1 https://github.com/automl/auto-scan.git /content/autoscan
# %cd /content/autoscan
# !pip install -e /content/autoscan


import functools
import logging

import numpy as np
import torch

import autoscan

logging.basicConfig(level=logging.INFO)

# ## What is Multi-Objective Optimization?
# Instead of one scalar, `evaluate_pipeline` returns several — often conflicting —
# objectives, and the optimizer searches for a *Pareto front*: configurations where
# no objective can improve without worsening another. AutoScAn's built-in MO optimizers
# are all bracket-based, like `asha`/`hyperband`, they require a
# `autoscan.Fidelity` parameter in the search space. This tutorial focuses on `PriMO`,
# the recommended choice when you also have expert priors to inject.

# ## The Optimization Task: ZDT1
# We use **ZDT1**, a standard
# multi-dimensional benchmark from the multi-objective optimization literature
# (Zitzler, Deb & Thiele). seed [this](https://pymoo.org/problems/multi/zdt.html) for more details.
N_DIMS = 3
MAX_FIDELITY = 10


def zdt1(x: list[float]) -> tuple[float, float]:
    """The true (noise-free) ZDT1 objectives."""
    f1 = x[0]
    g = 1 + 9 / (N_DIMS - 1) * sum(x[1:])
    f2 = g * (1 - np.sqrt(f1 / g))
    return f1, f2


# ## Setting Up the Search Space
# The search space is `N_DIMS` independent floats plus a fidelity parameter —
# PriMO reuses a multi-fidelity bracket internally, so the space needs a
# `autoscan.Fidelity` parameter even though `N_DIMS` itself has nothing to do with
# fidelity. `evaluate_pipeline` returns a list of objectives instead of a single
# float.


def zdt1_multi_fidelity(x: list[float], fidelity: int) -> tuple[float, float]:
    f1_true, f2_true = zdt1(x)
    noise_scale = 0.15 * (MAX_FIDELITY - fidelity) / MAX_FIDELITY
    f1 = f1_true + np.random.normal(0, noise_scale)
    f2 = f2_true + np.random.normal(0, noise_scale)
    return f1, f2


class ZDT1FidelitySpace(autoscan.PipelineSpace):
    """ZDT1 with a fidelity parameter, required by every AutoScAn MO optimizer."""

    x0 = autoscan.Float(0.0, 1.0)
    x1 = autoscan.Float(0.0, 1.0)
    x2 = autoscan.Float(0.0, 1.0)
    fidelity = autoscan.Fidelity(autoscan.Integer(1, MAX_FIDELITY))


def evaluate_pipeline(x0: float, x1: float, x2: float, fidelity: int) -> dict:
    f1, f2 = zdt1_multi_fidelity([x0, x1, x2], fidelity)
    return dict(objective_to_minimize=[f1, f2])


# ## Expert Priors per Objective, with PriMO
# PriMO takes per-objective expert priors, provided as a *prior center* (the
# configuration you expect to be good for that objective) and a *prior
# confidence* per parameter (0 = ignore the prior, 1 = trust it fully). This
# is the same idea as single-objective priors, just one set per objective instead
# of one overall.
#
# Here we encode exactly what we derived analytically about ZDT1: `f1` is minimized
# near `x0 = 0`, `f2` is minimized near `x0 = 1`, and both are minimized by keeping
# the remaining dimensions near `0`.

prior_points = {
    "prior_centers": {
        "good-f1": {"x0": 0.05, "x1": 0.1, "x2": 0.1},
        "good-f2": {"x0": 0.95, "x1": 0.1, "x2": 0.1},
    },
    "prior_confidences": {
        "good-f1": {"x0": 0.75, "x1": 0.75, "x2": 0.75},
        "good-f2": {"x0": 0.75, "x1": 0.75, "x2": 0.75},
    },
}

# ### Building the PriMO Optimizer
# Building the optimizer as a `partial` lets us thread the priors through while
# still keeping the conversion automatic:


def build_primo_optimizer(prior_points: dict):
    prior_centers = prior_points["prior_centers"]
    prior_confidences = prior_points["prior_confidences"]

    return functools.partial(
        autoscan.algorithms.primo,
        prior_centers=prior_centers,
        mo_selector="epsnet",
        prior_confidences=prior_confidences,
        initial_design_size=5,
        eta=2,
        cost_aware=False,
        device="cpu",
    )


# ### Run PriMO

autoscan.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_primo/",
    pipeline_space=ZDT1FidelitySpace(),
    worker_fidelities_to_spend=100,
    optimizer=build_primo_optimizer(prior_points),
    overwrite_root_directory=True,
)

# !python -m autoscan.status results_primo/

# PriMO starts with an MOASHA-driven initial design (`initial_design_size` full
# fidelity-ladder sweeps), `moasha` is also available on its own as a
# prior-free multi-objective bracket optimizer, e.g. `optimizer="moasha"`, then
# switches to Bayesian optimization once enough evaluations have accumulated,
# using the priors to bias sampling toward each objective's expected optimum
# while still exploring the trade-off surface between them.

# ## Inspecting the Pareto Front

import pandas as pd

df = pd.read_csv("results_primo/summary/full.csv")
print(df[["objective_to_minimize"]].head())

# Each row's `objective_to_minimize` holds both `[f1, f2]` values; the
# non-dominated subset of these across all evaluations is the discovered Pareto
# front — the more of the true `f2 = 1 - sqrt(f1)` curve it traces out, the better
# the search.

# ## Key Takeaways
# 1. **Multiple objectives**: return a list under `objective_to_minimize` instead
#    of a single float; AutoScAn then searches for a Pareto front, not one optimum.
# 2. **Fidelity is required for PriMO**: it reuses a multi-fidelity bracket
#    internally, so the space needs a `autoscan.Fidelity` parameter even in an
#    otherwise flat search.
# 3. **Priors are per-objective**: `prior_centers` / `prior_confidences` take one
#    entry per objective (or per belief you want to express), each a full
#    parameter -> value / parameter -> confidence mapping.

# For more advanced examples:
# - [Multi-Objective Example](https://github.com/automl/auto-scan/tree/master/autoscan_examples/efficiency)
# - [Efficiency Techniques Tutorial](https://colab.research.google.com/github/automl/auto-scan/blob/master/tutorials/3_efficiency_techniques.ipynb)
# - [Ask-and-Tell Interface](https://github.com/automl/auto-scan/tree/master/autoscan_examples/experimental)

# If you want to contribute new techniques or optimizers, check out the contribution guide [here](https://automl.github.io/auto-scan/latest/dev_docs/contributing/).
