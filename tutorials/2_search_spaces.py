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


# !git clone --depth 1 https://github.com/automl/neps.git /content/neps
# %cd /content/neps
# !pip install -e /content/neps


import neps
import logging

# ## Parameter Types
# NePS supports several parameter types for defining your search space.

# ### 1. Float Parameters
# Continuous parameters with optional log scaling.

# Linear scale
learning_rate_linear = neps.Float(lower=0.0001, upper=0.1)
# Log scale (useful for learning rates, regularization)
learning_rate_log = neps.Float(lower=1e-6, upper=1e-1, log=True)
# With prior value
dropout_rate = neps.Float(lower=0.0, upper=0.9, prior=0.5, prior_confidence="medium")

# ### 2. Integer Parameters
# Discrete integer parameters.

# Basic integer
num_layers = neps.Integer(lower=1, upper=10)
# With prior value
batch_size = neps.Integer(lower=16, upper=256, prior=64, prior_confidence="medium")
# With log scale
hidden_units = neps.Integer(lower=32, upper=2048, log=True)

# ### 3. Categorical Parameters
# Discrete choices with optional ordering.

# Basic categorical
optimizer = neps.Categorical(choices=["sgd", "adam", "adamw"])
# With prior value
activation = neps.Categorical(
    choices=["relu", "tanh", "sigmoid"], 
    prior=0,  # Index of default choice (relu)
    prior_confidence="high"
)

# ## Building Complex Search Spaces

# ### Using a Dictionary

# Simple dictionary-based search space
simple_space = dict(
    learning_rate=neps.Float(1e-6, 1e-1, log=True),
    batch_size=neps.Integer(16, 256),
    optimizer=neps.Categorical(["sgd", "adam"]),
)

# ### Using a PipelineSpace Class
# For complex search spaces, use the `PipelineSpace` class for conditioning one hyperparmater over other and better organization.

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

# ## Fidelity Parameters

# Use fidelity parameters for multi-fidelity optimization (train with different epochs, dataset sizes, etc.).

# Define a search space with a fidelity parameter
pipeline_space = dict(
    learning_rate=neps.Float(1e-6, 1e-1, log=True),
    optimizer=neps.Categorical(["sgd", "adam"]),
    # Fidelity: wrap the Integer or Float object in neps.Fidelity()
    epochs=neps.Fidelity(neps.Integer(1, 10)),
)

# ## Important: Constraint-Free Search Spaces
#
# For optimization algorithms to work effectively, your search space should be **completely searchable**
# without hidden constraints between hyperparameters.
#
# **Bad Example: Constrained search space**
# If there's an implicit constraint like `max_units % max_layers = 0 and max_units <= 8 * max_layers`:
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
# units_per_layer = neps.Integer(1, 8)  # Independent of max_layers
# # Now all combinations are valid and the optimizer can search freely!
# ```
#
# **Why this matters:**
# - Optimization algorithms assume all hyperparameter combinations in the space are valid
# - Hidden constraints can confuse the optimizer and reduce efficiency
# - A constraint-free space allows the algorithm to explore the entire search region
# - When you have interdependencies, reformulate the space to use independent dimensions

# ## Conditional Search Spaces

# You can also define conditional parameters that only apply under certain conditions.
# This is useful for architecture search where some parameters only apply to certain layer types.
#
# The important pattern is to define a reusable search grammar and call `.resample()`
# when the same grammar should be sampled independently in multiple places.

# Constructors for the conditional parts of the pipeline. In a real project these
# would create layers, modules, preprocessing steps, or optimizer objects.
def dense_layer(num_neurons: int, activation: str) -> dict:
    return {
        "layer_type": "dense",
        "num_neurons": num_neurons,
        "activation": activation,
    }


def conv_layer(num_filters: int, kernel_size: int) -> dict:
    return {
        "layer_type": "conv",
        "num_filters": num_filters,
        "kernel_size": kernel_size,
    }


def make_pipeline(*blocks: dict, learning_rate: float, optimizer: str) -> dict:
    return {
        "blocks": list(blocks),
        "learning_rate": learning_rate,
        "optimizer": optimizer,
    }


class ConditionalPipelineSpace(neps.PipelineSpace):
    """Build a variable pipeline from independently sampled conditional blocks.

    Each block independently chooses between a dense and convolutional branch. The
    branch-specific parameters are sampled as part of the selected operation, and
    the final evaluation receives the constructed pipeline object.
    """

    _dense_layer = neps.Operation(
        operator=dense_layer,
        kwargs={
            "num_neurons": neps.Integer(64, 512, log=True),
            "activation": neps.Categorical(["relu", "gelu", "elu"]),
        },
    )
    _conv_layer = neps.Operation(
        operator=conv_layer,
        kwargs={
            "num_filters": neps.Integer(16, 128, log=True),
            "kernel_size": neps.Categorical([3, 5, 7]),
        },
    )

    _block = neps.Categorical(
        choices=(
            _dense_layer,
            _conv_layer,
        ),
    )
    _learning_rate = neps.Float(1e-5, 1e-2, log=True)
    _optimizer = neps.Categorical(["adam", "adamw", "sgd"])

    pipeline = neps.Operation(
        operator=make_pipeline,
        args=(
            _block.resample(),
            _block.resample(),
            _block.resample(),
        ),
        kwargs={
            "learning_rate": _learning_rate,
            "optimizer": _optimizer,
        },
    )


def evaluate_conditional_pipeline(pipeline: dict) -> float:
    """Evaluate the fully constructed pipeline."""

    loss = 0.5

    for block in pipeline["blocks"]:
        if block["layer_type"] == "dense":
            loss -= 0.015 * (block["num_neurons"] / 512)
            loss -= 0.005 if block["activation"] == "relu" else 0
        else:
            loss -= 0.01 * (block["num_filters"] / 128)
            loss -= 0.005 if block["kernel_size"] == 3 else 0

    loss -= 0.02 if pipeline["optimizer"] == "adamw" else 0
    return max(0.1, loss)


conditional_space = ConditionalPipelineSpace()
neps.run(
    evaluate_pipeline=evaluate_conditional_pipeline,
    pipeline_space=conditional_space,
    root_directory="conditional_search_space_example/",
    evaluations_to_spend=5,
    optimizer="random_search",
    overwrite_root_directory=True,
)

# The serialized config stores the sampled resolution path, not just flat
# user-facing argument names.
#!cat conditional_search_space_example/configs/config_1/config.yaml

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
# If there are interdependencies between hyperparameters (e.g., max_units <= 8 * max_layers),
# refactor the space to use independent dimensions (e.g., units_per_layer and max_layers).
# Constraints between hyperparameters can confuse optimization algorithms and reduce efficiency.
#
# For more details, see the [NePS Documentation](https://automl.github.io/neps/latest/reference/neps_spaces/).

# Next steps:
# - Explore [**Efficiency Techniques**](https://colab.research.google.com/github/automl/neps/blob/master/tutorials/3_efficiency_techniques.ipynb) like multi-fidelity optimization.
