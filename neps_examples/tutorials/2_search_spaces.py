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

# # Defining Search Spaces in NePS
# This tutorial covers the different parameter types and how to define expressive search spaces for your optimization problems.

# ## Installation and Setup

# %%
# !pip install neural-pipeline-search

# %%
import neps
import logging

# ## Parameter Types
# NePS supports several parameter types for defining your search space.

# ### 1. Float Parameters
# Continuous parameters with optional log scaling.

# %%
# Linear scale
learning_rate_linear = neps.Float(lower=0.0001, upper=0.1)

# Log scale (useful for learning rates, regularization)
learning_rate_log = neps.Float(lower=1e-6, upper=1e-1, log=True)

# With prior value
dropout_rate = neps.Float(lower=0.0, upper=0.9, prior=0.5, prior_confidence="medium")

print("Float parameters created successfully")

# ### 2. Integer Parameters
# Discrete integer parameters.

# %%
# Basic integer
num_layers = neps.Integer(lower=1, upper=10)

# With prior value
batch_size = neps.Integer(lower=16, upper=256, prior=64, prior_confidence="medium")

# With log scale
hidden_units = neps.Integer(lower=32, upper=2048, log=True)

print("Integer parameters created successfully")

# ### 3. Categorical Parameters
# Discrete choices with optional ordering.

# %%
# Unordered categorical
optimizer = neps.Categorical(choices=["sgd", "adam", "adamw"])

# With prior/default value
activation = neps.Categorical(
    choices=["relu", "tanh", "sigmoid"], 
    prior=0,  # Index of default choice (relu)
    prior_confidence="high"
)

print("Categorical parameters created successfully")

# ## Building Complex Search Spaces

# ### Using a Dictionary

# %%
# Simple dictionary-based search space
simple_space = dict(
    learning_rate=neps.Float(1e-6, 1e-1, log=True),
    batch_size=neps.Integer(16, 256),
    optimizer=neps.Categorical(["sgd", "adam"]),
)

# ### Using a PipelineSpace Class
# For larger search spaces, use the `PipelineSpace` class for better organization.

# %%
class MyOptimizationSpace(neps.PipelineSpace):
    """Define a structured search space for neural architecture search."""
    
    # Architecture parameters
    num_layers = neps.Integer(lower=2, upper=6, prior=3, prior_confidence="medium")
    num_neurons = neps.Integer(lower=64, upper=512, log=True, prior=256, prior_confidence="medium")
    activation = neps.Categorical(
        choices=["relu", "elu", "gelu"],
        prior=0,
        prior_confidence="medium"
    )
    
    # Training hyperparameters
    learning_rate = neps.Float(lower=1e-6, upper=1e-1, log=True, prior=1e-3, prior_confidence="medium")
    optimizer = neps.Categorical(
        choices=["sgd", "adam", "adamw"],
        prior=1,
        prior_confidence="high"
    )
    
    # Regularization
    dropout_rate = neps.Float(lower=0.0, upper=0.9, prior=0.1, prior_confidence="medium")
    weight_decay = neps.Float(lower=0.0, upper=1e-2, log=True, prior=1e-4, prior_confidence="medium")

# %% [markdown]
# ## Fidelity Parameters

# %% [markdown]
# Use fidelity parameters for multi-fidelity optimization (train with different epochs, dataset sizes, etc.).

# %%
# Define a search space with a fidelity parameter
fidelity_space = dict(
    learning_rate=neps.Float(1e-6, 1e-1, log=True),
    optimizer=neps.Categorical(["sgd", "adam"]),
    # Fidelity: wrap Integer in neps.Fidelity()
    epochs=neps.Fidelity(neps.Integer(1, 10)),
)

print("Fidelity space created successfully")

# ## Important: Constraint-Free Search Spaces
#
# For optimization algorithms to work effectively, your search space should be **completely searchable**
# without hidden constraints between hyperparameters.
#
# **Bad Example: Constrained search space**
# If there's an implicit constraint like `max_units * max_layers <= 1024`:
# ```python
# max_layers = neps.Integer(1, 10)
# max_units = neps.Integer(64, 1024)
# # Problem: Not all combinations are valid! The optimizer might waste time exploring
# # configurations that violate the implicit constraint.
# ```
#
# **Good Example: Constraint-free search space**
# Refactor to use independent dimensions:
# ```python
# max_layers = neps.Integer(1, 10)
# units_per_layer = neps.Integer(64, 256)  # Independent of max_layers
# # Now all combinations are valid and the optimizer can search freely!
# ```
#
# **Why this matters:**
# - Optimization algorithms assume all hyperparameter combinations in the space are valid
# - Hidden constraints can confuse the optimizer and reduce efficiency
# - A constraint-free space allows the algorithm to explore the entire search region
# - When you have interdependencies, reformulate the space to use independent dimensions

# %% [markdown]
# ## Conditional Search Spaces

# %% [markdown]
# You can also define conditional parameters that only apply under certain conditions.
# This is useful for architecture search where some parameters only apply to certain layer types.

# %%
# Example: conditional parameter handling in the pipeline function
def conditional_pipeline(
    layer_type: str,
    num_neurons: int = None,
    kernel_size: int = None,
    **kwargs
) -> float:
    """Example showing how to handle conditional parameters."""
    
    if layer_type == "dense":
        # Only num_neurons matters for dense layers
        config = {"neurons": num_neurons}
    elif layer_type == "conv":
        # Only kernel_size matters for conv layers
        config = {"kernel_size": kernel_size}
    
    # Simulate some loss
    return 0.5

# %% [markdown]
# For proper conditional parameters, use dictionary-based spaces and handle them in your pipeline function.

# %% [markdown]
# ## Example: Complete Search Space

# %%
# A complete example with a real optimization
class ComprehensiveSpace(neps.PipelineSpace):
    """Complete example with multiple parameter types."""
    
    # Continuous hyperparameters
    learning_rate = neps.Float(1e-5, 1e-2, log=True, prior=1e-3, prior_confidence="high")
    weight_decay = neps.Float(1e-6, 1e-2, log=True, prior=1e-4, prior_confidence="medium")
    
    # Discrete architecture choices
    num_layers = neps.Integer(2, 8, prior=4, prior_confidence="medium")
    hidden_size = neps.Integer(64, 512, prior=256, prior_confidence="medium")
    
    # Categorical choices
    optimizer = neps.Categorical(
        ["adam", "adamw", "sgd"],
        prior=1,  # adamw
        prior_confidence="high"
    )
    activation = neps.Categorical(
        ["relu", "gelu", "elu"],
        prior=0,  # relu
        prior_confidence="medium"
    )

# Define pipeline function
def train_model(
    learning_rate: float,
    weight_decay: float,
    num_layers: int,
    hidden_size: int,
    optimizer: str,
    activation: str,
    **kwargs
) -> float:
    """Example training function that uses multiple parameter types."""
    
    # In practice, you'd build a model with these parameters
    # and return the validation loss
    
    # Simulated loss calculation
    loss = 0.5
    loss -= 0.05 * np.log10(learning_rate / 1e-3)  # Learning rate matters
    loss -= 0.02 * np.log10(hidden_size / 256)      # Bigger network better
    loss -= 0.01 if optimizer == "adamw" else 0     # adamw often better
    loss -= 0.01 if activation == "relu" else 0     # relu is standard
    
    return max(0.1, loss)

# Run a small optimization
import numpy as np

logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=train_model,
    pipeline_space=ComprehensiveSpace(),
    root_directory="search_space_example/",
    evaluations_to_spend=10,  # Quick demo
)

# %% [markdown]
# ## Key Points
#
# - **Float**: For continuous parameters, use `log=True` for log-scaled distributions
# - **Integer**: For discrete counts, also supports `log=True` for exponential spacing
# - **Categorical**: For discrete choices between multiple options
# - **Fidelity**: Wrap parameters with `neps.Fidelity()` for multi-fidelity optimization
# - **PipelineSpace**: Use class-based spaces for better organization in large search spaces
# - **Priors**: Use `prior` and `prior_confidence` to incorporate domain knowledge
#
# **Critical Design Principle**: Ensure your search space is **constraint-free**.
# If there are interdependencies between hyperparameters (e.g., max_units × max_layers ≤ 1024),
# refactor the space to use independent dimensions (e.g., units_per_layer and max_layers).
# Constraints between hyperparameters can confuse optimization algorithms and reduce efficiency.
#
# For more details, see the [NePS Documentation](https://automl.github.io/neps/latest/reference/neps_spaces/).
