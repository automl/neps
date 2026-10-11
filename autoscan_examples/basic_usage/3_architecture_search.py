"""
This example demonstrates neural architecture search using AutoScAn Spaces to define and
optimize PyTorch models. The search space consists of a 3-cell sequential architecture
where each cell contains a Conv2d layer followed by an activation function. The Conv2d
kernel size is sampled from integers in [2, 7], and the activation is chosen from
{ReLU, Sigmoid, Tanh}. Each cell independently samples its kernel size and activation,
allowing AutoScAn to explore diverse architectural configurations and find optimal designs.

Search Space Structure:
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

import numpy as np
import torch
import torch.nn as nn
import autoscan
import logging


# Define the AutoScAn space for the neural network architecture
# It reuses the same building blocks multiple times, with different sampled parameters.
class NN_Space(autoscan.PipelineSpace):

    # Parameters with prefixed _ are internal and will not be given to the evaluation
    # function
    _kernel_size = autoscan.Integer(2, 7)

    # Building blocks of the neural network architecture
    # The convolution layer with sampled kernel size
    _conv = autoscan.Operation(
        operator=nn.Conv2d,
        kwargs={
            "in_channels": 3,
            "out_channels": 3,
            "kernel_size": _kernel_size.resample(),
            "padding": "same",
        },
    )

    # Non-linearity layer sampled from a set of choices
    _nonlinearity = autoscan.Categorical(
        choices=(
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
        )
    )

    # A cell consisting of a convolution followed by a non-linearity
    _cell = autoscan.Operation(
        operator=nn.Sequential,
        args=(
            _conv.resample(),
            _nonlinearity.resample(),
        ),
    )

    # The full model consisting of three cells stacked sequentially
    # This will be given to the evaluation function as 'model'
    model = autoscan.Operation(
        operator=nn.Sequential,
        args=(
            _cell.resample(),
            _cell.resample(),
            _cell.resample(),
        ),
    )


# Defining the pipeline, using the model from the NN_space space as callable
def evaluate_pipeline(model: torch.nn.Module) -> float:
    x = torch.ones(size=[1, 3, 220, 220])
    result = np.sum(model(x).detach().numpy().flatten())

    return result


if __name__ == "__main__":
    # Run AutoScAn with the defined pipeline and space and show the best configuration
    pipeline_space = NN_Space()
    logging.basicConfig(level=logging.INFO)
    autoscan.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/architecture_search_example",
        worker_evaluations_to_spend=5,
    )
    autoscan.status(
        "results/architecture_search_example",
        print_summary=True,
    )
