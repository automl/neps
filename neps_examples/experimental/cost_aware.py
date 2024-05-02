import logging
import time

import numpy as np

import neps


def run_pipeline(
    pipeline_directory, float1, float2, categorical, integer1, integer2
):
    start = time.time()
    y = -float(np.sum([float1, float2, int(categorical), integer1, integer2]))
    end = time.time()
    return {
        "loss": y,
        "cost": (end - start) + float1,
    }


pipeline_space = dict(
    float1=neps.FloatParameter(lower=0, upper=1, log=False),
    float2=neps.FloatParameter(
        lower=0, upper=10, log=False, default=10, default_confidence="medium"
    ),
    categorical=neps.CategoricalParameter(choices=[0, 1]),
    integer1=neps.IntegerParameter(lower=0, upper=1, log=False),
    integer2=neps.IntegerParameter(lower=0, upper=1, log=False),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/cost_aware_example",
    searcher="cost_cooling",
    max_evaluations_total=12,  # TODO(Jan): remove
    initial_design_size=5,
    budget=100,
)
previous_results, pending_configs = neps.status("results/cost_aware_example")
