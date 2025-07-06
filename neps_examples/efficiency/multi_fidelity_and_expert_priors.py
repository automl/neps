import logging

import numpy as np
import neps

# This example demonstrates NePS uses both fidelity and expert priors to
# optimize hyperparameters of a pipeline.

def evaluate_pipeline(float1, float2, integer1, fidelity):
    objective_to_minimize = -float(np.sum([float1, float2, integer1])) / fidelity
    return objective_to_minimize


pipeline_space = dict(
    float1=neps.Float(
        lower=1,
        upper=1000,
        log=False,
        prior=600,
        prior_confidence="medium",
    ),
    float2=neps.Float(
        lower=-10,
        upper=10,
        prior=0,
        prior_confidence="medium",
    ),
    integer1=neps.Integer(
        lower=0,
        upper=50,
        prior=35,
        prior_confidence="low",
    ),
    fidelity=neps.Integer(
        lower=1,
        upper=10,
        is_fidelity=True,
    ),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/multifidelity_priors",
    fidelities_to_spend=25,  # For an alternate stopping method see multi_fidelity.py
)
