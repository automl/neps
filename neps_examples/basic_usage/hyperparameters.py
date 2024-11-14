import logging

import numpy as np

import neps


def run_pipeline(float1, float2, categorical, integer1, integer2):
    # In this example we maximize an arbitrary sum for demonstration purposes
    loss = -float(np.sum([float1, float2, int(categorical), integer1, integer2]))
    return loss


pipeline_space = dict(
    float1=neps.Float(lower=0, upper=1),
    float2=neps.Float(lower=-10, upper=10),
    categorical=neps.Categorical(choices=[0, 1]),
    integer1=neps.Integer(lower=0, upper=1),
    integer2=neps.Integer(lower=1, upper=1000, log=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/hyperparameters_example",
    post_run_summary=True,
    max_evaluations_total=15,
)
