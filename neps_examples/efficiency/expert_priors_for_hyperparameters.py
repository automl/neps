import logging
import time
from warnings import warn

import neps

def run_pipeline(some_float, some_integer, some_cat):
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning)
    return evaluate_pipeline(some_float, some_integer, some_cat)

def evaluate_pipeline(some_float, some_integer, some_cat):
    start = time.time()
    if some_cat != "a":
        y = some_float + some_integer
    else:
        y = -some_float - some_integer
    end = time.time()
    return {
        "objective_to_minimize": y,
        "info_dict": {
            "test_score": y,
            "train_time": end - start,
        },
    }


# neps uses the default values and a confidence in this default value to construct a prior
# that speeds up the search
pipeline_space = dict(
    some_float=neps.Float(
        lower=1, upper=1000, log=True, default=900, default_confidence="medium"
    ),
    some_integer=neps.Integer(
        lower=0, upper=50, default=35, default_confidence="low"
    ),
    some_cat=neps.Categorical(
        choices=["a", "b", "c"], default="a", default_confidence="high"
    ),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/user_priors_example",
    max_evaluations_total=15,
)
