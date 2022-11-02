import logging
import time

import neps


def run_pipeline(some_float, some_integer, some_cat):
    start = time.time()
    if some_cat != "a":
        y = some_float + some_integer
    else:
        y = -some_float - some_integer
    end = time.time()
    return {
        "loss": y,
        "info_dict": {
            "test_score": y,
            "train_time": end - start,
        },
    }


# neps uses the default values and a confidence in this default value to construct a prior
# that speeds up the search
pipeline_space = dict(
    some_float=neps.FloatParameter(
        lower=1, upper=1000, log=True, default=900, default_confidence="medium"
    ),
    some_integer=neps.IntegerParameter(
        lower=0, upper=50, default=35, default_confidence="low"
    ),
    some_cat=neps.CategoricalParameter(
        choices=["a", "b", "c"], default="a", default_confidence="high"
    ),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/user_priors_example",
    max_evaluations_total=15,
)
previous_results, pending_configs = neps.status("results/user_priors_example")
