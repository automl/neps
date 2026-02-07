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

# # Efficient Optimization with NePS: Multi-Fidelity and Advanced Techniques
# This tutorial covers advanced techniques to speed up hyperparameter optimization:
# multi-fidelity learning, expert priors, parallelization, and custom optimizers.

# ## Installation and Setup

# %%
# !pip install neural-pipeline-search

# %%
import neps
import logging
import numpy as np
import time

logging.basicConfig(level=logging.INFO)

# ## The Optimization Task
# Let's use a simulated deep learning task throughout this tutorial.

# %%
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
    optimizer_bonus = {"sgd": 0.0, "adamw": 0.05}.get(optimizer, 0)
    
    # Learning rate impact
    lr_penalty = -0.1 * np.log10(learning_rate / 0.01)
    
    # Network size impact
    neurons_impact = -0.02 * np.log2(num_neurons / 256)
    
    # Epoch impact (gets smaller as epochs increase)
    epoch_impact = -0.05 * np.sqrt(epochs / 10)
    
    # Simulated noisy loss
    loss = (
        base_loss 
        + optimizer_bonus 
        + lr_penalty 
        + neurons_impact 
        + epoch_impact
        + np.random.normal(0, 0.02)
    )
    
    return max(0.1, loss)

# %%
# ## Technique 1: Multi-Fidelity Optimization
# Train configurations at different fidelities (e.g., different epoch counts) 
# to efficiently explore the search space.

# ### Define Search Space with Fidelity

# %%
class MultiFidelitySpace(neps.PipelineSpace):
    """Search space for multi-fidelity optimization."""
    learning_rate = neps.Float(1e-6, 1e-1, log=True)
    optimizer = neps.Categorical(["sgd", "adamw"])
    num_neurons = neps.Integer(64, 1024)
    # Fidelity parameter: wrap Integer in neps.Fidelity()
    epochs = neps.Fidelity(neps.Integer(1, 10))

# %%
# ### Run Multi-Fidelity Optimization

# %%
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_multi_fidelity/",
    pipeline_space=MultiFidelitySpace(),
    evaluations_to_spend=15,
    overwrite_root_directory=True,
)

# %% [markdown]
# The optimizer intelligently samples configurations at low epochs (cheaper) 
# and high epochs (more accurate) to find good configurations efficiently.

# %%
# !python -m neps.status results_multi_fidelity/ --best_configs

# %%
# ## Technique 2: Incorporating Expert Priors
# Provide prior values and confidence levels to incorporate domain knowledge.

# ### Define Search Space with Priors

# %%
class ExpertPriorSpace(neps.PipelineSpace):
    """Search space with expert priors incorporated."""
    
    # We believe adamw with these settings is good (high confidence)
    learning_rate = neps.Float(
        1e-6, 1e-1, 
        log=True, 
        prior=0.001,  # Common prior
        prior_confidence="high"
    )
    
    optimizer = neps.Categorical(
        ["sgd", "adamw"],
        prior=1,  # adamw
        prior_confidence="high"
    )
    
    num_neurons = neps.Integer(
        64, 1024,
        prior=256,
        prior_confidence="low"  # Less confident about exact size
    )
    
    epochs = neps.Fidelity(neps.Integer(1, 10))

# %% [markdown]
# ### Run Optimization with Priors

# %%
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_with_priors/",
    pipeline_space=ExpertPriorSpace(),
    evaluations_to_spend=15,
    overwrite_root_directory=True,
)

# %% [markdown]
# The optimizer starts with your suggested configuration and uses it as a warm-start,
# potentially finding better solutions faster.

# %%
# !python -m neps.status results_with_priors/ --best_configs

# %% [markdown]
# ## Technique 3: Parallelization

# %% [markdown]
# NePS makes parallelization effortless. Multiple processes can work on the same 
# `root_directory` simultaneously.

# %% [markdown]
# ### Single Process Example (Sequential)

# %%
# Sequential run
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_sequential/",
    pipeline_space=ExpertPriorSpace(),
    evaluations_to_spend=6,
    overwrite_root_directory=True,
)

print("Sequential run complete")

# %% [markdown]
# ### Parallel Execution Pattern

# %% [markdown]
# To parallelize, run multiple `neps.run()` calls with the same `root_directory`:
#
# ```bash
# # In terminal, start multiple workers in parallel:
# python worker.py &
# python worker.py &
# python worker.py &
# 
# # They'll coordinate and divide work automatically
# ```
#
# Or continue an existing run with a higher budget:

# %%
# Continue the sequential run with more evaluations
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_sequential/",
    pipeline_space=ExpertPriorSpace(),
    evaluations_to_spend=12,  # Increase from 6 to 12
    overwrite_root_directory=True,
)

print("Extended run complete - 12 total evaluations")

# %%
# !python -m neps.status results_sequential/ --best_configs

# %% [markdown]
# ## Technique 4: Custom Optimizers

# %% [markdown]
# NePS supports multiple search algorithms. Explore available options:

# %%
from neps.optimizers.algorithms import OptimizerChoice, PredefinedOptimizers
from neps.utils.common import extract_keyword_defaults

print("Available optimization algorithms:")
for algo in OptimizerChoice:
    print(f"  - {algo}")

# %% [markdown]
# ### Using Different Optimizers

# %%
# Get available hyperparameters for an optimizer
print("\nBayesian Optimization hyperparameters:")
print(extract_keyword_defaults(PredefinedOptimizers.get("bayesian_optimization")))

# %%
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_custom_optimizer/",
    pipeline_space=ExpertPriorSpace(),
    evaluations_to_spend=10,
    optimizer="asha",  # Use Async ASHA
    overwrite_root_directory=True,
)

# %% [markdown]
# ## Technique 5: Combining Strategies

# %% [markdown]
# Combine multiple techniques for maximum efficiency.

# %%
class CombinedSearchSpace(neps.PipelineSpace):
    """Combines multi-fidelity, priors, and complex search space."""
    
    # Architecture with priors
    num_layers = neps.Integer(2, 6, prior=3, prior_confidence="medium")
    num_neurons = neps.Integer(64, 512, log=True, prior=256, prior_confidence="low")
    activation = neps.Categorical(
        ["relu", "elu", "gelu"],
        prior=0,  # relu
        prior_confidence="high"
    )
    
    # Training with priors
    learning_rate = neps.Float(1e-5, 1e-2, log=True, prior=1e-3, prior_confidence="high")
    optimizer = neps.Categorical(
        ["adam", "adamw", "sgd"],
        prior=1,  # adamw
        prior_confidence="high"
    )
    
    # Regularization with priors
    dropout_rate = neps.Float(0.0, 0.5, prior=0.1, prior_confidence="medium")
    weight_decay = neps.Float(1e-6, 1e-2, log=True, prior=1e-4, prior_confidence="medium")
    
    # Multi-fidelity (epochs)
    epochs = neps.Fidelity(neps.Integer(2, 20))

# %% [markdown]
# Run optimization with combined strategies:

# %%
start_time = time.time()

neps.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_combined/",
    pipeline_space=CombinedSearchSpace(),
    evaluations_to_spend=20,
    optimizer="asha",
    overwrite_root_directory=True,
)

elapsed = time.time() - start_time
print(f"\nOptimization completed in {elapsed:.2f} seconds")

# %%
# !python -m neps.status results_combined/ --best_configs

# %% [markdown]
# ## Comparing Strategies

# %% [markdown]
# Let's visualize how the best loss improves over time for different approaches:

# %%
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Optimization Strategies Comparison", fontsize=14, fontweight="bold")

strategies = [
    ("Sequential", "results_sequential/summary/full.csv"),
    ("With Priors", "results_with_priors/summary/full.csv"),
    ("Custom Optimizer", "results_custom_optimizer/summary/full.csv"),
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

# %% [markdown]
# ## Key Takeaways

# %% [markdown]
# 1. **Multi-Fidelity**: Use cheaper proxies (fewer epochs) to quickly filter bad configs
# 2. **Expert Priors**: Warm-start with good defaults and confidence levels
# 3. **Parallelization**: Simply run multiple processes with the same `root_directory`
# 4. **Custom Optimizers**: Choose from various search algorithms
# 5. **Combination**: Stack techniques for maximum efficiency
#
# These techniques can reduce optimization time by 10-50% compared to standard random search!

# %% [markdown]
# For more advanced examples:
# - [Multi-Objective Optimization](https://github.com/automl/neps/tree/master/neps_examples/efficiency)
# - [Ask-and-Tell Interface](https://github.com/automl/neps/tree/master/neps_examples/experimental)
# - [Architecture Search](https://github.com/automl/neps/tree/master/neps_examples/basic_usage)
