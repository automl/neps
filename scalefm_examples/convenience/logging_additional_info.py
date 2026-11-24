import logging
import time
from warnings import warn

import numpy as np

import scalefm


def evaluate_pipeline(float1, float2, categorical, integer1, integer2):
    start = time.time()
    objective_to_minimize = -float(
        np.sum([float1, float2, int(categorical), integer1, integer2])
    )
    end = time.time()
    return {
        "objective_to_minimize": objective_to_minimize,
        "cost": 1.0,  # The training cost of this run, e.g. its FLOPs
        "info_dict": {  # Optionally include additional information as an info_dict
            "train_time": end - start,
        },
    }


class HPOSpace(scalefm.PipelineSpace):
    float1 = scalefm.Float(lower=0, upper=1)
    float2 = scalefm.Float(lower=-10, upper=10)
    categorical = scalefm.Categorical(choices=(0, 1))
    integer1 = scalefm.Integer(lower=0, upper=1)
    integer2 = scalefm.Integer(lower=1, upper=1000, log=True)


logging.basicConfig(level=logging.INFO)
scalefm.acquire(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=HPOSpace(),
    root_directory="results/logging_additional_info",
    worker_evaluations_to_spend=5,
)
