"""
This example demonstrates the full capabilities of NePS Spaces
by defining a neural network architecture using PyTorch modules.
It showcases how to interact with the NePS Spaces API to create,
sample and evaluate a neural network pipeline.
It also demonstrates how to convert the pipeline to a callable
and how to run NePS with the defined pipeline and space.
"""

import numpy as np
import torch
import torch.nn as nn
import neps


# Define the NEPS space for the neural network architecture
# It reuses the same building blocks multiple times, with different sampled parameters.
class NN_Space(neps.PipelineSpace):

    _kernel_size = neps.Integer(2, 7)

    # Regular parameter (not prefixed with _) - will be sampled and shown in results
    learning_rate = neps.Float(0.0001, 0.01, log=True)
    optimizer_name = neps.Categorical(["adam", "sgd", "rmsprop"])

    _conv = neps.Operation(
        operator=nn.Conv2d,
        kwargs={
            "in_channels": 3,
            "out_channels": 3,
            "kernel_size": neps.Resampled(_kernel_size),
            "padding": "same",
        },
    )

    _nonlinearity = neps.Categorical(
        choices=(
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
        )
    )

    _cell = neps.Operation(
        operator=nn.Sequential,
        args=(
            neps.Resampled(_conv),
            neps.Resampled(_nonlinearity),
        ),
    )

    model = neps.Operation(
        operator=nn.Sequential,
        args=(
            neps.Resampled(_cell),
            neps.Resampled(_cell),
            neps.Resampled(_cell),
        ),
    )


# Defining the pipeline, using the model from the NN_space space as callable
def evaluate_pipeline(model: nn.Sequential, learning_rate: float, optimizer_name: str):
    x = torch.ones(size=[1, 3, 220, 220])
    result = np.sum(model(x).detach().numpy().flatten())
    # Use learning_rate and optimizer_name in a simple way to show they're being passed
    optimizer_multiplier = {"adam": 1.0, "sgd": 1.1, "rmsprop": 0.9}.get(
        optimizer_name, 1.0
    )
    return result * learning_rate * optimizer_multiplier


# Run NePS with the defined pipeline and space and show the best configuration
pipeline_space = NN_Space()
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/architecture_search_example",
    evaluations_to_spend=5,
    overwrite_root_directory=True,
)
neps.status(
    "results/architecture_search_example",
    print_summary=True,
)
