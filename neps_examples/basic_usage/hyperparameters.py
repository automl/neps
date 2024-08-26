import logging
import time

import numpy as np
import math
import random

import neps


def run_pipeline(float1, float2, float3, categorical, integer1, integer2):
    loss = -float(
        np.sum(
            [
                (float1 * float2 / (float3 + 1)) * int(categorical),
                integer1,
                math.log(integer2),
            ]
        )
    )  # Random noise
    # time.sleep(0.7)  # For demonstration purposes
    return loss


pipeline_space = dict(
    float1=neps.FloatParameter(lower=0, upper=1),
    float2=neps.FloatParameter(lower=0, upper=20),
    float3=neps.FloatParameter(lower=0, upper=5),
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
    max_evaluations_total=50,
)
