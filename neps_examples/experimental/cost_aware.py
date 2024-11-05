import logging
import time
from warnings import warn

import numpy as np

import neps


def run_pipeline(float1, float2, categorical, integer1, integer2):
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning)
    return evaluate_pipeline(float1, float2, categorical, integer1, integer2)

def evaluate_pipeline(
    float1, float2, categorical, integer1, integer2
):
    start = time.time()
    y = -float(np.sum([float1, float2, int(categorical), integer1, integer2]))
    end = time.time()
    return {
        "loss": y,
        "cost": (end - start) + float1,
    }


pipeline_space = dict(
    float1=neps.Float(lower=0, upper=1, log=False),
    float2=neps.Float(
        lower=0, upper=10, log=False, default=10, default_confidence="medium"
    ),
    categorical=neps.Categorical(choices=[0, 1]),
    integer1=neps.Integer(lower=0, upper=1, log=False),
    integer2=neps.Integer(lower=0, upper=1, log=False),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/cost_aware_example",
    searcher="cost_cooling",
    max_evaluations_total=12,  # TODO(Jan): remove
    initial_design_size=5,
    budget=100,
)
previous_results, pending_configs = neps.status("results/cost_aware_example")
