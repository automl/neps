import logging
import time

import numpy as np

import neps


def run_pipeline(  # pylint: disable=unused-argument
    config, config_working_directory, previous_working_directory
):
    x = np.array(config.get_hps(), dtype=float)
    start = time.time()
    y = -float(np.sum(x))
    end = time.time()

    return {
        "loss": y,
        "info_dict": {
            "config_id": None,
            "val_score": y,
            "test_score": y,
            "train_time": end - start,
        },
    }


if __name__ == "__main__":
    pipeline_space = dict(
        x1=neps.FloatParameter(lower=0, upper=1, log=False),
        x2=neps.FloatParameter(lower=0, upper=1, log=False),
        x3=neps.FloatParameter(lower=0, upper=1, log=False),
        x4=neps.FloatParameter(lower=0, upper=1, log=False),
        x5=neps.FloatParameter(lower=0, upper=1, log=False),
        x6=neps.CategoricalParameter(choices=[0, 1]),
        x7=neps.CategoricalParameter(choices=[0, 1]),
        x8=neps.CategoricalParameter(choices=[0, 1]),
        x9=neps.CategoricalParameter(choices=[0, 1]),
        x10=neps.CategoricalParameter(choices=[0, 1]),
    )

    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        working_directory="results/hyperparameters_example",
        n_iterations=20,
        hp_kernels=["m52", "hm"],
        use_new_metahyper=True,
    )
    previous_results, pending_configs, pending_configs_free = neps.read_results(
        "results/hyperparameters_example"
    )

    # print("Best found configuration: ", id2config[incumbent]["config"])
    print(f"A total of {len(previous_results)} unique configurations were evaluated.")
