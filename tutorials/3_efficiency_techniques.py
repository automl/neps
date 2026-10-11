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

# # Efficient Optimization with AutoScAn: Multi-Fidelity and Advanced Techniques
# This tutorial covers advanced techniques to speed up hyperparameter optimization:
# multi-fidelity learning, expert priors, optimizer selection, and parallelization.

# ## Installation and Setup


# !git clone --depth 1 https://github.com/automl/auto-scan.git /content/autoscan
# %cd /content/autoscan
# !pip install -e /content/autoscan


import autoscan
import logging
import numpy as np
import time

logging.basicConfig(level=logging.INFO)

# ## The Optimization Task
# Let's use a simulated deep learning task throughout this tutorial.

def evaluate_pipeline(
    learning_rate: float,
    optimizer: str,
    num_neurons: int,
    epochs: int = 10,
    **kwargs
) -> float:
    """Simulated training function.

    In practice, this trains a real neural network and returns validation loss.
    """
    np.random.seed(42)

    # Simulate training: loss decreases with epochs
    base_loss = 0.8

    # Optimizer impact
    optimizer_bonus = {"sgd": 0.0, "adamw": -0.05, "adam": -0.03}.get(optimizer, 0)

    # Learning rate impact
    lr_penalty = -0.1 * np.log10(learning_rate / 0.01)

    # Network size impact
    neurons_impact = -0.02 * np.log2(num_neurons / 256)

    # Optional architecture/regularization parameters used in later examples
    layers_impact = -0.01 * (kwargs.get("num_layers", 3) - 3)
    activation_impact = -0.01 if kwargs.get("activation", "relu") == "relu" else 0.0
    dropout_penalty = 0.03 * kwargs.get("dropout_rate", 0.0)
    weight_decay_penalty = 0.01 * np.log10(kwargs.get("weight_decay", 1e-4) / 1e-4)

    # Epoch impact (gets smaller as epochs increase)
    epoch_impact = -0.05 * np.sqrt(epochs / 10)

    # Simulated noisy loss
    loss = (
        base_loss
        + optimizer_bonus
        + lr_penalty
        + neurons_impact
        + layers_impact
        + activation_impact
        + dropout_penalty
        + weight_decay_penalty
        + epoch_impact
        + np.random.normal(0, 0.02)
    )

    return max(0.1, loss)

# ## Technique 1: Multi-Fidelity Optimization
# Train configurations at different fidelities (e.g., different epoch counts)
# to efficiently explore the search space.
#
# Multi-fidelity optimizers available in AutoScAn include:
# - `successive_halving`: synchronous promotion through one bracket
# - `asha`: asynchronous Successive Halving, useful with parallel workers
# - `hyperband`: multiple bracket layouts, the default for flat HPO spaces with fidelity
# - `async_hb`: asynchronous Hyperband
# - `ifbo`: freeze-thaw/in-context optimizer for learning-curve style fidelities
# - `priorband`: Hyperband-style multi-fidelity search with expert priors
# - `moasha`, `mo_hyperband`, `primo`: multi-objective variants
#
# For complex `PipelineSpace` objects, AutoScAn also exposes native variants such as
# `autoscan_hyperband` and `autoscan_priorband`, and `optimizer="auto"` chooses them when
# the search space contains fidelity parameters.

# ### Define Search Space with Fidelity

class MultiFidelitySpace(autoscan.PipelineSpace):
    """Search space for multi-fidelity optimization."""
    learning_rate = autoscan.Float(1e-6, 1e-1, log=True)
    optimizer = autoscan.Categorical(["sgd", "adamw"])
    num_neurons = autoscan.Integer(64, 1024)
    # Fidelity parameter: Use IntegerFidelity
    epochs = autoscan.IntegerFidelity(1, 10)


# ### Run Multi-Fidelity Optimization

autoscan.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_multi_fidelity/",
    pipeline_space=MultiFidelitySpace(),
    worker_fidelities_to_spend=60,
    overwrite_root_directory=True,
    optimizer="asha",  # Use ASHA, designed for multi-fidelity workloads
)

# The optimizer intelligently samples configurations at low epochs (cheaper)
# and high epochs (more accurate) to find good configurations efficiently.
# In real training code, include `pipeline_directory` and
# `previous_pipeline_directory` in `evaluate_pipeline` so promoted configurations can
# load checkpoints from earlier fidelity rungs.

# !python -m autoscan.status results_multi_fidelity/

# ## Technique 2: Incorporating Expert Priors
# Provide prior values and confidence levels to incorporate domain knowledge.

# ### Define Search Space with Priors

class ExpertPriorSpace(autoscan.PipelineSpace):
    """Search space with expert priors incorporated."""

    # We believe adamw with these settings is good (high confidence)
    learning_rate = autoscan.Float(
        1e-6, 1e-1,
        log=True,
        prior=0.001,  # Common prior
        prior_confidence="high"
    )

    optimizer = autoscan.Categorical(
        ["sgd", "adamw"],
        prior=1,  # adamw
        prior_confidence="high"
    )

    num_neurons = autoscan.Integer(
        64, 1024,
        prior=256,
        prior_confidence="low"  # Less confident about exact size
    )

    epochs = autoscan.IntegerFidelity(1, 10)

# ### Run Optimization with Priors

autoscan.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_with_priors/",
    pipeline_space=ExpertPriorSpace(),
    worker_fidelities_to_spend=60,
    optimizer="priorband",
    overwrite_root_directory=True,
)

# `priorband` combines the fidelity ladder with prior-guided sampling. If you leave
# `optimizer="auto"`, AutoScAn chooses the appropriate prior-aware multi-fidelity
# optimizer for spaces that contain both priors and fidelities.

# !python -m autoscan.status results_with_priors/

# ## Technique 3: Optimizer Selection

# AutoScAn supports multiple search algorithms. You can let `optimizer="auto"` pick from
# the search-space structure, or specify an optimizer explicitly.

from autoscan.optimizers.algorithms import PredefinedOptimizers
from autoscan.utils.common import extract_keyword_defaults

print("Available optimization algorithms:")
for algo in sorted(PredefinedOptimizers):
    print(f"  - {algo}")

# ### Multi-Fidelity Optimizer Parameters

for algo in ["successive_halving", "asha", "hyperband", "async_hb", "ifbo", "priorband"]:
    print(f"\n{algo} hyperparameters:")
    print(extract_keyword_defaults(PredefinedOptimizers[algo]))

autoscan.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_custom_optimizer/",
    pipeline_space=MultiFidelitySpace(),
    worker_fidelities_to_spend=60,
    optimizer="async_hb",
    overwrite_root_directory=True,
)

# ## Technique 4: Parallelization

# AutoScAn makes parallelization effortless. Multiple processes can work on the same
# `root_directory` simultaneously.

# ### Single Process Example (Sequential)

# Sequential run
autoscan.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_sequential/",
    pipeline_space=ExpertPriorSpace(),
    worker_evaluations_to_spend=6,
    overwrite_root_directory=True,
)

# !python -m autoscan.status results_sequential/

print("Sequential run complete")

# ### Parallel Execution Pattern

# To parallelize, run multiple `autoscan.run()` calls with the same `root_directory`:

import multiprocessing as mp
def run_worker():
    autoscan.run(
        evaluate_pipeline=evaluate_pipeline,
        root_directory="results_parallel/",
        pipeline_space=ExpertPriorSpace(),
        worker_evaluations_to_spend=6, # each worker's budget
        overwrite_root_directory=False,  # workers attach, don't reset
    )
# Launch 3 workers in parallel
processes = [mp.Process(target=run_worker) for _ in range(3)]
for p in processes:
    p.start()
for p in processes:
    p.join()

# They'll coordinate and sample/run new configurations without conflicts, all writing to the same results directory.

# !python -m autoscan.status results_parallel/

# ## Technique 5: Combining Strategies

# Combine multiple techniques for maximum efficiency.
class CombinedSearchSpace(autoscan.PipelineSpace):
    """Combines multi-fidelity, priors, and complex search space."""

    # Architecture with priors
    num_layers = autoscan.Integer(2, 6, prior=3, prior_confidence="medium")
    num_neurons = autoscan.Integer(64, 512, log=True, prior=256, prior_confidence="low")
    activation = autoscan.Categorical(
        ["relu", "elu", "gelu"],
        prior=0,  # relu
        prior_confidence="high"
    )
    # Training with priors
    learning_rate = autoscan.Float(1e-5, 1e-2, log=True, prior=1e-3, prior_confidence="high")
    optimizer = autoscan.Categorical(
        ["adam", "adamw", "sgd"],
        prior=1,  # adamw
        prior_confidence="high"
    )

    # Regularization with priors
    dropout_rate = autoscan.Float(0.0, 0.5, prior=0.1, prior_confidence="medium")
    weight_decay = autoscan.Float(1e-6, 1e-2, log=True, prior=1e-4, prior_confidence="medium")

    # Multi-fidelity (epochs)
    epochs = autoscan.IntegerFidelity(2, 20)

# Run optimization with combined strategies:

start_time = time.time()

autoscan.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_combined/",
    pipeline_space=CombinedSearchSpace(),
    worker_fidelities_to_spend=80,
    optimizer="priorband",
    overwrite_root_directory=True,
)

elapsed = time.time() - start_time
print(f"\nOptimization completed in {elapsed:.2f} seconds")

# !python -m autoscan.status results_combined/

# ## Comparing Strategies

# Let's visualize how the best loss improves over time for different approaches:

import pandas as pd
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Optimization Strategies Comparison", fontsize=14, fontweight="bold")
strategies = [
    ("Sequential", "results_sequential/summary/full.csv"),
    ("With Priors", "results_with_priors/summary/full.csv"),
    ("Async Hyperband", "results_custom_optimizer/summary/full.csv"),
    ("Combined", "results_combined/summary/full.csv"),
]
for idx, (name, path) in enumerate(strategies):
    ax = axes[idx // 2, idx % 2]

    try:
        df = pd.read_csv(path)
        # Calculate incumbent trajectory
        df['best_loss'] = df['objective_to_minimize'].cummin()

        ax.plot(df.index, df['best_loss'], 'o-', linewidth=2, markersize=5)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Best Loss")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    except FileNotFoundError:
        ax.text(0.5, 0.5, "Results not available", ha='center', va='center')
        ax.set_title(name)
plt.tight_layout()
plt.show()

# ## Key Takeaways
# 1. **Multi-Fidelity**: Use cheaper proxies (fewer epochs) to quickly filter bad configs
# 2. **Expert Priors**: Warm-start with good defaults and confidence levels
# 3. **Parallelization**: Simply run multiple processes with the same `root_directory`
# 4. **Optimizer Selection**: Choose from various search algorithms, or use `auto`
# 5. **Combination**: Stack techniques for maximum efficiency
#
# These techniques can reduce optimization time by 10-50% compared to standard random search!

# Next steps:
# - Explore [**Multi-Objective Optimization**](https://colab.research.google.com/github/automl/auto-scan/blob/master/tutorials/4_multi_objective.ipynb) like PriMO Algorithm.

# If you want to contribute new techniques or optimizers, check out the contribution guide [here](https://automl.github.io/auto-scan/latest/dev_docs/contributing/).
