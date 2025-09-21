import logging
import numpy as np
import neps
import socket
import os
# This example demonstrates how to use NePS to optimize hyperparameters
# of a pipeline. The pipeline is a simple function that takes in
# five hyperparameters and returns their sum.
# Neps uses the default optimizer to minimize this objective function.

def evaluate_pipeline(float1, float2, categorical, integer1, integer2):
    objective_to_minimize = -float(
        np.sum([float1, float2, int(categorical), integer1, integer2])
    )
    return objective_to_minimize


class HPOSpace(neps.PipelineSpace):
    float1 = neps.Float(min_value=0, max_value=1)
    float2 = neps.Float(min_value=-10, max_value=10)
    categorical = neps.Categorical(choices=(0, 1))
    integer1 = neps.Integer(min_value=0, max_value=1)
    integer2 = neps.Integer(min_value=1, max_value=1000, log=True)


logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=HPOSpace(),
    root_directory="results/hyperparameters_example",
    evaluations_to_spend=30,
    worker_id=f"worker_1-{socket.gethostname()}-{os.getpid()}",
)
