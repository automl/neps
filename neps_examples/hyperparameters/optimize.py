import logging
import time

import numpy as np

import neps


def run_pipeline(working_directory, float1, float2, categorical, integer1, integer2):
    start = time.time()
    y = -float(np.sum([float1, float2, int(categorical), integer1, integer2]))
    end = time.time()
    return {
        "loss": y,
        "info_dict": {
            "test_score": y,
            "train_time": end - start,
        },
    }


if __name__ == "__main__":
    pipeline_space = dict(
        float1=neps.FloatParameter(lower=0, upper=1, log=False),
        float2=neps.FloatParameter(lower=0, upper=10, log=False),
        categorical=neps.CategoricalParameter(choices=[0, 1]),
        integer1=neps.IntegerParameter(lower=0, upper=1, log=False),
        integer2=neps.IntegerParameter(lower=0, upper=1, log=False),
    )
    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        working_directory="results/hyperparameters_example",
        n_iterations=20,
        hp_kernels=["m52", "hm"],
    )
    previous_results, pending_configs, pending_configs_free = neps.read_results(
        "results/hyperparameters_example"
    )

    print(f"A total of {len(previous_results)} unique configurations were evaluated.")
