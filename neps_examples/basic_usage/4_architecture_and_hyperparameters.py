"""
This example demonstrates how to combine neural network architecture
search with hyperparameter optimization using NePS.
"""

import neps
import torch
import numpy as np
from torch import nn

# Using the space from the architecture search example
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


# Extend the architecture search space with a hyperparameter
extended_space = NN_Space().add(neps.Integer(16, 128), name="batch_size")

def evaluate_pipeline(model: torch.nn.Module, batch_size: int) -> float:
    # For demonstration, we return a dummy objective value
    # In practice, you would train and evaluate the model here
    x = torch.ones(size=[1, 3, 220, 220])
    result = np.sum(model(x).detach().numpy().flatten())

    objective_value = batch_size * result  # Dummy computation
    return objective_value


if __name__ == "__main__":
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=extended_space,
        root_directory="results/architecture_with_hp_example",
        evaluations_to_spend=5,
        overwrite_root_directory=True,
    )
    neps.status(
        root_directory="results/architecture_with_hp_example",
        print_summary=True,
    )
