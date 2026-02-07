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

# # Getting Started with NePS: Basic HPO
# Welcome! This tutorial introduces you to **Hyperparameter Optimization (HPO)** with NePS.
# We'll start with simple synthetic functions, then move to a real deep learning task.

# ## Installation and Setup

# %%capture
# !pip install neural-pipeline-search

# ## Quick Start: Optimizing a Synthetic Function
# Let's start with a simple optimization problem to understand NePS basics.

# %%
import math
import neps
import logging

logging.basicConfig(level=logging.INFO)

# The task: minimize the 2D Branin function
def branin(x1, x2):
    """The 2-dimensional Branin function.
    
    Reference: https://www.sfu.ca/~ssurjano/branin.html
    """
    a = 1
    b = 5.1 / (4 * math.pi**2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / 8 * math.pi

    f = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
    return f

# %%
# Define the search space
class MyTestSpace(neps.PipelineSpace):
    x1=neps.Float(-5, 10)
    x2=neps.Float(0, 15)

# Run the optimization
neps.run(
    pipeline_space=MyTestSpace(),
    root_directory="branin_demo/",
    evaluations_to_spend=25,
    evaluate_pipeline=branin,
    optimizer="random_search",
)

# Check the optimization results:
# %%capture
# !tail ./branin_demo/best_config_trajectory.txt

# Great! NePS found the minimum loss over 25 evaluations. 
# This demonstrates the basic workflow:
# 1. Define an objective function (`branin`)
# 2. Specify a search space
# 3. Call `neps.run()` to optimize

# ## Running a Basic HPO on a Real Deep Learning Task
# Now let's optimize hyperparameters for an actual neural network.

# ### Step 1: Define the Training Pipeline

# %%
import numpy as np
from typing import Optional

# For demo purposes, we create a simple synthetic training function
def evaluate_pipeline(
    learning_rate: float,
    optimizer: str,
    num_neurons: int,
    **kwargs
) -> float:
    """Mock training function that simulates a deep learning pipeline.
    
    In practice, this would train a real model.
    """
    # Simulate training: create a synthetic loss that decreases with good hyperparams
    np.random.seed(42)
    
    # Optimizer impact
    optimizer_bonus = {"sgd": 0.0, "adamw": 0.02}.get(optimizer, 0)
    
    # Learning rate impact (log scale)
    lr_impact = -0.1 * np.log10(learning_rate / 0.01)
    
    # Network size impact
    neurons_impact = -0.01 * np.log2(num_neurons / 256)
    
    # Simulated noisy loss
    loss = 0.5 + optimizer_bonus + lr_impact + neurons_impact + np.random.normal(0, 0.02)
    loss = max(0.1, loss)  # Ensure positive loss
    
    return loss

# ### Step 2: Define the Search Space

# Define hyperparameters to optimize
class MySpace(neps.PipelineSpace):
    learning_rate=neps.Float(1e-6, 1e-1, log=True)
    optimizer=neps.Categorical(["sgd", "adamw"])
    num_neurons=neps.Integer(64, 1024)

# ### Step 3: Run NePS Optimization

# %%
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_hpo_basic/",
    pipeline_space=MySpace(),
    evaluations_to_spend=10,
    optimizer="random_search",
    overwrite_root_directory=True,
)

# ### Analyzing the Results
# Check the status of your run:

# %%capture
# !python -m neps.status results_hpo_basic/ --best_configs

# The `root_directory` contains:
# - `configs/`: Individual configurations and their evaluations
# - `summary/`: Summary of all trials, contaning best configurations and their trajectory.

# %%
import pandas as pd

# Load and inspect the results
df = pd.read_csv("results_hpo_basic/summary/full.csv")
print("\nTop 3 configurations by objective_to_minimize:")
print(df.nsmallest(3, "objective_to_minimize")[["objective_to_minimize", "config.learning_rate", "config.optimizer", "config.num_neurons"]])

# ## Key Takeaways
# 1. **NePS Pattern**: Define function → Define space → Call `neps.run()`
# 2. **Search Space Types**: `Float`, `Integer`, `Categorical` parameters
# 3. **Optimization Output**: Tracked in `root_directory` with easy analysis tools
# 4. **Scalability**: Same API works for complex deep learning tasks
#
# Next steps:
# - Learn about **Multi-Fidelity Optimization** (faster convergence)
# - Explore **Expert Priors** (incorporate domain knowledge)
# - Try **Parallelization** (distributed optimization)

# For more examples and documentation:
# - [NePS Examples](https://github.com/automl/neps/tree/master/neps_examples)
# - [NePS Documentation](https://automl.github.io/neps/latest/)
