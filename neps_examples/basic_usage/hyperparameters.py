import logging
import time

import numpy as np
import math
import random

import neps

PRINT = False


def run_pipeline(float1, float2, float3, integer1, integer2):
    if PRINT:
        print("float1:", float1)
        print("float2:", float2)
        print("float3:", float3)
        # print("categorical:", categorical)
        print("integer1:", integer1)
        print("integer2:", integer2)
    loss = -float(
        integer2
        * np.sum(
            [
                (float1 * float2 / (float3 + 1)),  # * (int(categorical) + 1),
                integer1,
            ]
        )
    )  # Random noise
    # time.sleep(0.7)  # For demonstration purposes
    return {"loss": loss, "cost": float(integer2)}


pipeline_space = dict(
    float1=neps.FloatParameter(lower=0, upper=1, default=0.95),
    float2=neps.FloatParameter(lower=0, upper=20, default=19.5),
    float3=neps.FloatParameter(lower=0, upper=5, default=0.5),
    # categorical=neps.CategoricalParameter(choices=[0, 1]),
    integer1=neps.IntegerParameter(lower=0, upper=1, default=1),
    integer2=neps.IntegerParameter(lower=1, upper=1000, log=True, default=950),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    searcher="bayesian_optimization",
    pipeline_space=pipeline_space,
    root_directory="results/hyperparameters_example",
    post_run_summary=True,
    max_evaluations_total=50,
    use_prior=True,
)
