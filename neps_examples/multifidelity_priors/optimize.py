import logging
import time

import numpy as np

import neps


def run_pipeline(working_directory, float1, float2, categorical, integer1, fidelity):
    start = time.time()
    loss = -float(np.sum([float1, float2, int(categorical), integer1])) / fidelity
    end = time.time()
    return {
        "loss": loss,
        "info_dict": {  # Optionally include additional information
            "test_score": loss,
            "train_time": end - start,
        },
    }


pipeline_space = dict(
    float1=neps.FloatParameter(
        lower=1, upper=1000, log=False, default=600, default_confidence="medium"
    ),
    float2=neps.FloatParameter(lower=-10, upper=10),
    categorical=neps.CategoricalParameter(choices=[0, 1]),
    integer1=neps.IntegerParameter(
        lower=0, upper=50, default=35, default_confidence="low"
    ),
    fidelity=neps.IntegerParameter(lower=1, upper=5, is_fidelity=True),
)

searcher = "multifidelity_tpe"
searcher_kwargs = dict(use_priors=False, initial_design_size=7)

logging.basicConfig(level=logging.INFO)
searcher_output = "multifidelity_priors"
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory=f"results/{searcher_output}",
    max_evaluations_total=50,
    searcher=searcher,
    **searcher_kwargs,
)
previous_results, pending_configs = neps.status(f"results/{searcher_output}")
