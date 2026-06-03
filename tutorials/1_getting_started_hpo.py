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
# This tutorial introduces **Hyperparameter Optimization (HPO)** with NePS, starting with synthetic functions and progressing to real deep learning tasks.

# ## Installation
# Requires Python 3.10+. Install NePS via:

# %%
# !pip install neural-pipeline-search

# ## Example 1: Synthetic Function Optimization
# We start with a simple optimization problem to understand NePS basics.

# %%
import math
import neps
import logging

logging.basicConfig(level=logging.INFO)

def branin(x1, x2):
    """Minimize the 2D Branin function.
    
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
# Define the search space and run optimization
class BraninSpace(neps.PipelineSpace):
    x1 = neps.Float(-5, 10)
    x2 = neps.Float(0, 15)

neps.run(
    pipeline_space=BraninSpace(),
    root_directory="branin_demo/",
    evaluations_to_spend=25,
    evaluate_pipeline=branin,
    optimizer="random_search",
)

# Check the optimization results:
#!tail ./branin_demo/summary/best_config.txt

# Great! NePS found the minimum loss over 25 evaluations. 
# The NePS workflow always follows this pattern:
# 1. **Define** an objective function
# 2. **Specify** a search space
# 3. **Call** `neps.run()` to optimize

# ## Example 2: Deep Learning HPO
# Now we optimize hyperparameters for a neural network on MNIST (multi-class classification with CNNs).
#
# For the full training code, see [train.py](https://github.com/automl/neps/blob/master/tutorials/train.py).

# %%
from tutorials.train import training_pipeline
from tutorials.utils import set_seeds

# Test the training pipeline with a small subset
training_pipeline(
    # checkpoint controller
    out_dir=None,
    load_dir=None,
    # possible hyperparameters
    num_layers=1,
    batch_size=512,
    # possible fidelities
    subsample=0.1,
    epochs=2,
    val_fraction=0.3,
    allow_checkpointing=False,
    verbose=True,
)

# Note: `objective_to_minimize = 1 - val_accuracy` (validation error), since NePS minimizes.

# ### Running the HPO
# Now we apply the standard NePS pattern to optimize the training pipeline.

# Step 1: Create a NePS wrapper
def evaluate_pipeline(
    pipeline_directory,  # For saving checkpoints
    previous_pipeline_directory,  # For loading checkpoints
    **hyperparameters,  # Determined by HPO algorithms and passed by NePS
):
    return training_pipeline(
        **hyperparameters,
        out_dir=pipeline_directory,
        load_dir=previous_pipeline_directory,
        # misc settings
        batch_size=512,
        log_neps_tensorboard=True,
        verbose=False,
        allow_checkpointing=True,
        use_for_demo=True  # toggle as desired
    )

# Step 2: Define the search space (tuning learning rate)
class HPODemoSpace(neps.PipelineSpace):
    learning_rate = neps.Float(1e-6, 1e-1, log=True)
pipeline_space = HPODemoSpace()

# Step 3: Run optimization
logging.basicConfig(level=logging.INFO, force=True)
set_seeds(1)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_hpo_demo",
    pipeline_space=pipeline_space,
    evaluations_to_spend=3  # HPO budget
)

# ### Analyzing Results
# Check the status and outputs:

!python -m neps.status results_hpo_demo/

# %%
# View the summary files:
!cat results_hpo_demo/summary/short.csv

!cat results_hpo_demo/summary/best_config_trajectory.txt

!cat results_hpo_demo/summary/best_config.txt

import pandas as pd

df = pd.read_csv("results_hpo_demo/summary/full.csv")
df.head()

# Commented out IPython magic to ensure Python compatibility.
# Load the tensorboard
# %load_ext tensorboard

# %tensorboard --logdir /content/neps/results_hpo_demo

# %% [markdown]
# For more details on the Tensorboard integration, see documentation [here](https://automl.github.io/neps/latest/reference/analyse/#visualizing-results).
#
# ## A slightly more elaborate HPO
#
# Now, we construct a slightly more elaborate search space and perform a longer HPO run.
#
# <!-- *NOTE*: If using a GPU or with time in hand, we recommend `run_pipeline()` or else `run_pipeline_demo()` for shorter, quicker runs. -->
#
# <!-- *NOTE, again*: Using `run_pipeline_demo()` may affect the multi-fidelity runs in this notebook. -->


class DemoSpace(neps.PipelineSpace):
    learning_rate = neps.Float(1e-6, 1e-1, log=True)
    optimizer = neps.Categorical(["sgd", "adamw"])
    num_neurons = neps.Integer(64, 1024)
pipeline_space = DemoSpace()

set_seeds(1)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    root_directory="results_hpo",
    pipeline_space=pipeline_space,
    evaluations_to_spend=8,
)

!python -m neps.status results_hpo/

# ## Key Takeaways
# 1. **NePS Pattern**: Define function → Define space → Call `neps.run()`
# 2. **search space Types**: `Float`, `Integer`, `Categorical` parameters
# 3. **Optimization Output**: Tracked in `root_directory` with easy analysis tools
# 4. **Scalability**: Same API works for complex deep learning tasks
#
# Next steps:
# - Learn more about [**Search Spaces**](https://colab.research.google.com/github/automl/neps/blob/master/tutorials/2_search_spaces.ipynb) and how to construct them for complex pipelines.
# - Explore [**Efficiency Techniques**](https://colab.research.google.com/github/automl/neps/blob/master/tutorials/3_efficiency_techniques.ipynb) like multi-fidelity optimization.
