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
from neps import (
    PipelineSpace,
    Operation,
    Categorical,
    Resampled,
)
from neps.space.neps_spaces import neps_space


# Define the neural network architecture using PyTorch as usual
class ReLUConvBN(nn.Module):
    def __init__(self, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.LazyConv2d(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=2,
                bias=False,
            ),
            nn.LazyBatchNorm2d(affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# Define the NEPS space for the neural network architecture
class NN_Space(PipelineSpace):
    _id = Operation(operator=Identity)
    _three = Operation(
        operator=nn.Conv2d,
        kwargs={
            "in_channels": 3,
            "out_channels": 3,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
    )
    _one = Operation(
        operator=nn.Conv2d,
        kwargs={
            "in_channels": 3,
            "out_channels": 3,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
        },
    )
    _reluconvbn = Operation(
        operator=ReLUConvBN,
        kwargs={"out_channels": 3, "kernel_size": 3, "stride": 1, "padding": 1},
    )

    _O = Categorical(choices=(_three, _one, _id))

    _C_ARGS = Categorical(
        choices=(
            (Resampled(_O),),
            (Resampled(_O), Resampled("model"), _reluconvbn),
            (Resampled(_O), Resampled("model")),
            (Resampled("model"),),
        ),
    )
    _C = Operation(
        operator=nn.Sequential,
        args=Resampled(_C_ARGS),
    )

    _model_ARGS = Categorical(
        choices=(
            (Resampled(_C),),
            (_reluconvbn,),
            (Resampled("model"),),
            (Resampled("model"), Resampled(_C)),
            (Resampled(_O), Resampled(_O), Resampled(_O)),
            (
                Resampled("model"),
                Resampled("model"),
                Resampled(_O),
                Resampled(_O),
                Resampled(_O),
                Resampled(_O),
                Resampled(_O),
                Resampled(_O),
            ),
        ),
    )
    model = Operation(
        operator=nn.Sequential,
        args=Resampled(_model_ARGS),
    )


# Defining the pipeline, using the model from the NN_space space as callable
def evaluate_pipeline(model: nn.Sequential):
    x = torch.ones(size=[1, 3, 220, 220])
    result = np.sum(model(x).detach().numpy().flatten())
    return result


# Run NePS with the defined pipeline and space and show the best configuration
pipeline_space = NN_Space()
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    optimizer=neps.algorithms.neps_random_search,
    root_directory="results/neps_spaces_nn_example",
    evaluations_to_spend=5,
    overwrite_root_directory=True,
)
neps.status(
    "results/neps_spaces_nn_example",
    print_summary=True,
    pipeline_space=pipeline_space,
)
