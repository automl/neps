import logging
import time

import neps


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
class HPOSpace(neps.PipelineSpace):
    some_float = (
        neps.Float(
            min_value=1,
            max_value=1000,
            log=True,
            prior=900,
            prior_confidence="medium",
        ),
    )
    some_integer = (
        neps.Integer(
            min_value=0,
            max_value=50,
            prior=35,
            prior_confidence="low",
        ),
    )
    some_cat = (
        neps.Categorical(
            choices=("a", "b", "c"),
            prior=0,
            prior_confidence="high",
        ),
    )


logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=HPOSpace(),
    root_directory="results/user_priors_example",
    max_evaluations_total=15,
)
