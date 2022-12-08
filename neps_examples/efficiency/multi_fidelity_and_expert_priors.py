import logging

import numpy as np

import neps


def run_pipeline(float1, float2, integer1, fidelity):
    loss = -float(np.sum([float1, float2, integer1])) / fidelity
    return loss


pipeline_space = dict(
    float1=neps.FloatParameter(
        lower=1, upper=1000, log=False, default=600, default_confidence="medium"
    ),
    float2=neps.FloatParameter(
        lower=-10, upper=10, default=0, default_confidence="medium"
    ),
    integer1=neps.IntegerParameter(
        lower=0, upper=50, default=35, default_confidence="low"
    ),
    fidelity=neps.IntegerParameter(lower=1, upper=10, is_fidelity=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/multifidelity_priors",
    max_evaluations_total=25,  # For an alternate stopping method see multi_fidelity.py
)
