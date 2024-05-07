import logging
import time

import numpy as np

import neps


def run_pipeline(float1, float2, categorical, integer1, integer2):
    loss = -float(np.sum([float1, float2, int(categorical), integer1, integer2]))
    return loss


pipeline_space = dict(
    float1=neps.FloatParameter(lower=0, upper=1),
    float2=neps.FloatParameter(lower=-10, upper=10),
    categorical=neps.CategoricalParameter(choices=[0, 1]),
    integer1=neps.IntegerParameter(lower=0, upper=1),
    integer2=neps.IntegerParameter(lower=1, upper=1000, log=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/hyperparameters_example",
    post_run_summary=True,
    max_evaluations_total=15,
)
