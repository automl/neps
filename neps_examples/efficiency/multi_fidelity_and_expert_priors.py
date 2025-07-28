import logging

import numpy as np
import neps

# This example demonstrates NePS uses both fidelity and expert priors to
# optimize hyperparameters of a pipeline.


def evaluate_pipeline(float1, float2, integer1, fidelity):
    objective_to_minimize = -float(np.sum([float1, float2, integer1])) / fidelity
    return objective_to_minimize


class HPOSpace(neps.PipelineSpace):
    float1 = neps.Float(
        min_value=1,
        max_value=1000,
        log=False,
        prior=600,
        prior_confidence="medium",
    )
    float2 = neps.Float(
        min_value=-10,
        max_value=10,
        prior=0,
        prior_confidence="medium",
    )
    integer1 = neps.Integer(
        min_value=0,
        max_value=50,
        prior=35,
        prior_confidence="low",
    )
    fidelity = neps.Fidelity(neps.Integer(min_value=1, max_value=10))


logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=HPOSpace(),
    root_directory="results/multifidelity_priors",
    max_evaluations_total=25,  # For an alternate stopping method see multi_fidelity.py
)
