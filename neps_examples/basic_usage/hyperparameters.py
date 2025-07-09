import logging
import numpy as np
import neps

def evaluate_pipeline(float1, float2, categorical, integer1, integer2):
    objective_to_minimize = -float(
        np.sum([float1, float2, int(categorical), integer1, integer2])
    )
    return objective_to_minimize


class PipelineSpace(neps.Pipeline):
    float1=neps.Float(min_value=0, max_value=1)
    float2=neps.Float(min_value=-10, max_value=10)
    categorical=neps.Categorical(choices=(0, 1))
    integer1=neps.Integer(min_value=0, max_value=1)
    integer2=neps.Integer(min_value=1, max_value=1000, log=True)

logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=PipelineSpace(),
    root_directory="results/hyperparameters_example",
    post_run_summary=True,
    max_evaluations_total=30,
)
