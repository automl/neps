import logging
import time

import neps


def run_pipeline(working_directory, some_float, some_cat):
    start = time.time()
    y = -some_float
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
        some_float=neps.FloatParameter(
            lower=0, upper=1, default=0.3, default_confidence=0.5
        ),
        some_cat=neps.CategoricalParameter(
            choices=["a", "b", "c"], default="a", default_confidence=0.3
        ),
    )
    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        working_directory="results/user_priors_example",
        max_evaluations_total=20,
    )
    previous_results, pending_configs, pending_configs_free = neps.read_results(
        "results/user_priors_example"
    )

    print(f"A total of {len(previous_results)} unique configurations were evaluated.")
