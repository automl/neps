"""
This example demonstrates joint optimization of neural architecture and hyperparameters
using NePS Spaces. The search space includes: (1) a 3-cell sequential architecture with
Conv2d layers (kernel size sampled from [2, 7]) and activations chosen from {ReLU,
Sigmoid, Tanh}, and (2) a batch_size hyperparameter sampled from integers in [16, 128].
NePS simultaneously optimizes both architectural choices and training hyperparameters.

Search Space Structure:
    batch_size: <sampled from [16, 128]>
    model: Sequential(
        Cell_1: Sequential(
            Conv2d(kernel_size=<sampled from [2, 7]>, ...),
            <sampled from {ReLU, Sigmoid, Tanh}>
        ),
        Cell_2: Sequential(
            Conv2d(kernel_size=<sampled from [2, 7]>, ...),
            <sampled from {ReLU, Sigmoid, Tanh}>
        ),
        Cell_3: Sequential(
            Conv2d(kernel_size=<sampled from [2, 7]>, ...),
            <sampled from {ReLU, Sigmoid, Tanh}>
        )
    )
"""

import neps
import torch
import numpy as np
from torch import nn
import logging


# Using the space from the architecture search example
class NN_Space(neps.PipelineSpace):

    # Integer Hyperparameter for the batch size
    batch_size = neps.Integer(16, 128)

    # Parameters with prefixed _ are internal and will not be given to the evaluation
    # function
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
    # This will be given to the evaluation function as 'model'
    model = neps.Operation(
        operator=nn.Sequential,
        args=(
            neps.Resampled(_cell),
            neps.Resampled(_cell),
            neps.Resampled(_cell),
        ),
    )


def evaluate_pipeline(model: torch.nn.Module, batch_size: int) -> float:
    # For demonstration, we return a dummy objective value
    # In practice, you would train and evaluate the model here
    x = torch.ones(size=[1, 3, 220, 220])
    result = np.sum(model(x).detach().numpy().flatten())

    objective_value = batch_size * result  # Dummy computation
    return objective_value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=NN_Space(),
        root_directory="results/architecture_with_hp_example",
        evaluations_to_spend=5,
    )
    neps.status(
        root_directory="results/architecture_with_hp_example",
        print_summary=True,
    )
