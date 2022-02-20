import logging
import time

import neps


def run_pipeline(working_directory, previous_working_directory, some_float, some_integer):
    start = time.time()
    y = -some_float + some_integer
    end = time.time()
    return {
        "loss": y,
        "info_dict": {
            "test_score": y,
            "train_time": end - start,
        },
    }


pipeline_space = dict(
    some_float=neps.FloatParameter(lower=0, upper=1),
    some_integer=neps.IntegerParameter(lower=1, upper=10, is_fidelity=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    working_directory="results/multi_fidelity_example",
    max_evaluations_total=20,
)
previous_results, pending_configs = neps.status("results/multi_fidelity_example")
