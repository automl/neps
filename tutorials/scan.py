# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # ScAn: Bayesian Optimization Aimed at the Target Compute
#
# The [ScaleFM demo](scalefm_demo.ipynb) acquired runs with Chinchilla's
# approach 1 and random search, and extrapolated them to a target compute. **ScAn**
# spends the same budget more deliberately: it tunes the model size and data together
# with the other training hyperparameters, and picks the runs that tell it most about
# the best loss at the target compute.
#
# <p align="center"><img src="static/scan_loss_over_cost.png" width="420"><br><b>ScAn</b>: the Gaussian process of its runs, extrapolated to the target</p>
#
# It runs on the same TabPFN surrogate benchmark, pipeline and search space as the
# demo.

# ## Installation
# Requires Python 3.11+. In Colab, run the cell below; locally, see the
# [README](README.md).

# !git clone --depth 1 https://github.com/automl/scalefm.git /content/scalefm_repo
# %cd /content/scalefm_repo/tutorials
# !pip install -e /content/scalefm_repo
# CARBS is not on PyPI, so it is installed from GitHub
# !pip install "carbs @ git+https://github.com/imbue-ai/carbs@8f7fee08658ad567d9dd6b6233247bc37d9962ea"
# Surrogate benchmark of TabPFN, it makes every run take a fraction of a second
# !pip install scan-bench

# +
import logging
import os

# The surrogates run in forked worker processes later on, which needs a single
# OpenMP thread (set before they load)
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
from pathlib import Path

from IPython.display import Image, display
from scan_bench import TabPFNTarget

import scalefm
import tabpfn_scan_bench as tabpfn

logging.basicConfig(level=logging.WARNING)
logging.getLogger("scalefm").setLevel(logging.INFO)
warnings.filterwarnings("ignore")


def show(*images, width=480):
    """Show PNG files, e.g. the plots of a run's `summary/`."""
    for image in images:
        display(Image(filename=image, width=width))


# -

# ## The pipeline
#
# `evaluate_pipeline` is the function `scalefm.acquire` calls with every config. It
# trains a model and returns what the scaling analysis needs: the loss to minimize, the
# training cost, the model size and data, and optionally downstream performances and
# the inference cost.
#
# Here, the surrogate benchmark of `scan_bench` stands in for the training: it predicts
# what a TabPFN training run would reach, its validation loss and its negative
# log-likelihood on real tabular datasets, a **downstream** metric.

# +
loss_benchmark = tabpfn.get_benchmark(TabPFNTarget.VAL_LOSS)
nll_benchmark = tabpfn.get_benchmark(TabPFNTarget.NLL)  # Fitted on first use, ~20s


def evaluate_pipeline(**config) -> dict:
    """Train TabPFN with `config`, here: ask the surrogates what it would reach."""
    run = tabpfn.build_config(**config)
    flops_per_cell = tabpfn.flops_estimator(**{**config, "total_cells": 1})
    return {
        "objective_to_minimize": loss_benchmark.query(run)["predictions"]["mean"],
        "cost": flops_per_cell * run.total_cells,  # Training FLOPs
        "inference_cost": flops_per_cell / 3,  # FLOPs of a forward pass over one cell
        "n_params": tabpfn.parameters_estimator(**config),
        "n_data": run.total_cells,
        "downstream_performances": {
            "real_nll": nll_benchmark.query(run)["predictions"]["mean"]
        },
    }


# -

# ## The search space
#
# The search space is the most important design decision in a study. It defines every
# parameter of a run: its type, its scale (log, linear, ...), its range and its
# constraints. If it is badly defined, no optimizer can make up for it. Mark the
# parameters that drive compute cost with `is_compute_scaling=True`: the scaling
# optimizers lay out their runs along these axes.
#
# The model size is determined by `embedding_size` and `num_layers`, the amount of data by
# `total_cells`.

class TabPFNSpace(scalefm.PipelineSpace):
    # Compute-scaling parameters: the model size and the data
    total_cells = scalefm.Integer(2**20, 2**35, log=True, log_base=2, is_compute_scaling=True)
    embedding_size = scalefm.Integer(4, 256, log=True, log_base=2, is_compute_scaling=True)
    num_layers = scalefm.Integer(2, 32, log=True, log_base=2, is_compute_scaling=True)
    # The other hyperparameters
    effective_batch_size = scalefm.Integer(16, 256, log=True, log_base=2)
    lr = scalefm.Float(1e-4, 5e-2, log=True)


# Every acquisition gets the same FLOP budget, at most a tenth of it per run, and we
# predict at a target 20x the cost of the largest run.

# +
RESULTS = Path("results/scaling_tutorial")

MAX_COST_PER_RUN = 2.8e14
BUDGET = MAX_COST_PER_RUN * 10
TARGET = 5.6e15
TARGETS = [TARGET]
# -

# ## Acquisition with ScAn
#
# ScAn is Bayesian optimization with scaling-specific parts: an information-theoretic
# acquisition aimed at `target_cost`, and a power-law prior mean. Unlike a static
# layout such as Chinchilla's, it fits a model to the runs before choosing the next
# one, which takes time, and needs earlier results to choose well, so by default it runs
# one run after the other. It also runs on parallel workers, as in section 5 of the
# demo, with the same `optimizer`.

# +
SCAN = RESULTS / "scan"
SCAN_OPTIMIZER = (
    "bayesian_optimization",
    {
        "acquisition_func_type": "IT-SCAN-JES",
        "target_cost": TARGET,
        "prior_mean": "power_law",
        "budget_schedule": "left_budget_contraction",
    },
)

tabpfn.set_seed(0)
scalefm.acquire(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=TabPFNSpace(),
    root_directory=SCAN,
    optimizer=SCAN_OPTIMIZER,
    cost_estimator=tabpfn.flops_estimator,
    max_cost_per_config=MAX_COST_PER_RUN,
    total_cost_to_spend=BUDGET,
    overwrite_root_directory=True,
)

full, _ = scalefm.status(SCAN)
print(f"{len(full)} runs, {full['cost'].sum():.2e} FLOPs,"
      f" best loss {full['objective_to_minimize'].min():.3f}")
# -

# ## Prediction with `gp_posterior`
#
# ScAn's own predictor, `gp_posterior`, which `predictor="auto"` picks, fits the same
# Gaussian process the optimizer used to all the runs. It needs the estimators to turn
# a predicted config into its model size and cost:

(prediction,) = scalefm.predict(
    SCAN, target_costs=TARGETS,
    params_estimator=tabpfn.parameters_estimator, cost_estimator=tabpfn.flops_estimator,
    cache_dir=SCAN / "gp_cache",
)
print(f"C={TARGET:.1e}: loss {prediction.loss:.3f},"
      f" N*={prediction.n_params:.2e}, D*={prediction.n_data:.2e}")
print("config:", prediction.config)
show(SCAN / "summary/pre_train/loss_over_cost.png",
     SCAN / "summary/downstream/real_nll/real_nll_over_cost.png")

# ScAn tunes every hyperparameter, and the Gaussian process predicts all of them, so
# each has a fit and a prediction at the target:

show(*[SCAN / f"summary/pre_train/parameters/{name}_over_cost.png"
       for name in ["n_params", "n_data", "embedding_size", "num_layers", "total_cells",
                    "lr", "effective_batch_size"]],
     width=420)

# And the predicted config at the target:

print((SCAN / "summary/pre_train/predictions_over_cost.txt").read_text())

# ## Chinchilla approach 1, for comparison
#
# The same budget spent by Chinchilla's approach 1, as in the demo, on the space with a
# fixed training recipe. `chinchilla3` predicts the loss, but no config, so we predict
# the config from the same runs with `power_law`:

# +
RECIPE = dict(num_layers=2, effective_batch_size=64, lr=1e-3)


class TabPFNChinchillaSpace(scalefm.PipelineSpace):
    total_cells = scalefm.Integer(2**20, 2**35, log=True, log_base=2, is_compute_scaling=True)
    embedding_size = scalefm.Integer(4, 256, log=True, log_base=2, is_compute_scaling=True)
    # The fixed recipe
    num_layers = RECIPE["num_layers"]
    effective_batch_size = RECIPE["effective_batch_size"]
    lr = RECIPE["lr"]


CHINCHILLA1 = RESULTS / "chinchilla1"

tabpfn.set_seed(0)
scalefm.acquire(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=TabPFNChinchillaSpace(),
    root_directory=CHINCHILLA1,
    optimizer=(
        "chinchilla",
        {
            "strategy": "approach1",
            "data_parameter_name": "total_cells",
            "params_estimator": tabpfn.parameters_estimator,
            "data_multipliers": [300, 1e3, 3e3, 1e4],
        },
    ),
    cost_estimator=tabpfn.flops_estimator,
    total_cost_to_spend=BUDGET,
    overwrite_root_directory=True,
)

(chinchilla_loss,) = scalefm.predict(CHINCHILLA1, target_costs=TARGETS)
(chinchilla_full,) = scalefm.predict(CHINCHILLA1, target_costs=TARGETS, predictor="power_law")
print(f"C={TARGET:.1e}: loss {chinchilla_loss.loss:.3f}")
print("config:", chinchilla_full.config)
# -

# ## Does the prediction hold?
#
# On the benchmark we can check a prediction: train the predicted config and compare its
# loss with the predicted one. The reference is the surrogate's own best config at the
# target, which `tabpfn.reference_optimum` finds by searching the benchmark directly.

# +
import matplotlib.pyplot as plt
import pandas as pd

reference = tabpfn.reference_optimum(TARGET)
quality = []


def check(acquisition, predicted_loss, config):
    """Train the predicted `config` and keep its loss next to the predicted one."""
    result = evaluate_pipeline(**config)
    quality.append({
        "acquisition": acquisition,
        "config FLOPs": f"{result['cost']:.2e}",
        "predicted loss": round(predicted_loss, 3),
        "evaluated loss": round(result["objective_to_minimize"], 3),
        "reference loss": round(reference.loss, 3),
        **{name: config[name] for name in
           ["embedding_size", "num_layers", "total_cells", "lr", "effective_batch_size"]},
    })
    return pd.DataFrame(quality).set_index("acquisition")


def plot_quality():
    """The predicted and evaluated loss of each predicted config, against the reference."""
    table = pd.DataFrame(quality)
    x = range(len(table))
    fig, ax = plt.subplots(figsize=(max(6.4, 2 * len(table) + 1), 4.4))
    ax.vlines(x, table["predicted loss"], table["evaluated loss"], color="#B0B0B0",
              linewidth=2, zorder=1)
    ax.scatter(x, table["predicted loss"], s=90, facecolors="white", edgecolors="#2B2B2B",
               linewidths=2, zorder=3, label="predicted")
    ax.scatter(x, table["evaluated loss"], s=90, color="#2B2B2B", zorder=3,
               label="evaluated")
    ax.axhline(reference.loss, color="#D55E00", linestyle="--", linewidth=1.5,
               label="reference")
    for i, row in table.iterrows():
        for column, color in [("evaluated loss", "#2B2B2B"), ("predicted loss", "#52514E")]:
            ax.annotate(f"{row[column]:.3f}", (i, row[column]), textcoords="offset points",
                        xytext=(10, -4), color=color)
    ax.set_xticks(list(x), table["acquisition"])
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.set(ylabel="validation loss", title=f"Loss of the predicted config at {TARGET:.1e} FLOPs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E5DF")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
              fontsize=9)
    plt.tight_layout()
    plt.show()


check("Chinchilla", chinchilla_loss.loss, {**RECIPE, **chinchilla_full.config})
check("ScAn", prediction.loss, prediction.config)
# -

plot_quality()
