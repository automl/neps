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

    # Building blocks of the neural network architecture
    # The convolution layer with sampled kernel size
    _conv = neps.Operation(
        operator=nn.Conv2d,
        kwargs={
            "in_channels": 3,
            "out_channels": 3,
            "kernel_size": neps.Resampled(_kernel_size),
            "padding": "same",
        },
    )

    # Non-linearity layer sampled from a set of choices
    _nonlinearity = neps.Categorical(
        choices=(
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
        )
    )

    # A cell consisting of a convolution followed by a non-linearity
    _cell = neps.Operation(
        operator=nn.Sequential,
        args=(
            neps.Resampled(_conv),
            neps.Resampled(_nonlinearity),
        ),
    )

    # The full model consisting of three cells stacked sequentially
    model = neps.Operation(
        operator=nn.Sequential,
        args=(
            neps.Resampled(_cell),
            neps.Resampled(_cell),
            neps.Resampled(_cell),
        ),
    )


# Defining the pipeline, using the model from the NN_space space as callable
def evaluate_pipeline(model: torch.nn.Module) -> float:
    x = torch.ones(size=[1, 3, 220, 220])
    result = np.sum(model(x).detach().numpy().flatten())

    return result


if __name__ == "__main__":
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
