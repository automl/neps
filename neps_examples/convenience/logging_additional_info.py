import logging
import time
from warnings import warn

import numpy as np

import neps


def evaluate_pipeline(float1, float2, categorical, integer1, integer2):
    start = time.time()
    objective_to_minimize = -float(
        np.sum([float1, float2, int(categorical), integer1, integer2])
    )
    end = time.time()
    return {
        "objective_to_minimize": objective_to_minimize,
        "info_dict": {  # Optionally include additional information as an info_dict
            "train_time": end - start,
        },
    }


class HPOSpace(neps.PipelineSpace):
    float1 = neps.Float(min_value=0, max_value=1)
    float2 = neps.Float(min_value=-10, max_value=10)
    categorical = neps.Categorical(choices=(0, 1))
    integer1 = neps.Integer(min_value=0, max_value=1)
    integer2 = neps.Integer(min_value=1, max_value=1000, log=True)


logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=HPOSpace(),
    root_directory="results/logging_additional_info",
    evaluations_to_spend=5,
)
