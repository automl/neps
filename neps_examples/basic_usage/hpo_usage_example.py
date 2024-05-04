import logging
import time

import numpy as np

import neps

def run_pipeline(
    float_name1,
    float_name2,
    categorical_name1,
    categorical_name2,
    integer_name1,
    integer_name2,
):
    # neps optimize to find values that maximizes sum, for demonstration only
    loss = -float(
        np.sum(
            [float_name1, float_name2, categorical_name1, integer_name1, integer_name2]
        )
    )
    if categorical_name2 == "a":
        loss += 1

    return loss


logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space="search_space_example.yaml",
    root_directory="results/hyperparameters_example",
    post_run_summary=True,
    max_evaluations_total=15,
)
