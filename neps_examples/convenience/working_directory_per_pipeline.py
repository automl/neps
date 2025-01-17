import logging
from pathlib import Path
from warnings import warn

import numpy as np

import neps


def evaluate_pipeline(pipeline_directory: Path, float1, categorical, integer1):
    # When adding pipeline_directory to evaluate_pipeline, neps detects its presence and
    # passes a directory unique for each pipeline configuration. You can then use this
    # pipeline_directory to create / save files pertaining to a specific pipeline, e.g.:
    weight_file = pipeline_directory / "weight_file.txt"
    weight_file.write_text("0")

    objective_to_minimize = -float(np.sum([float1, int(categorical), integer1]))
    return objective_to_minimize


pipeline_space = dict(
    float1=neps.Float(lower=0, upper=1),
    categorical=neps.Categorical(choices=[0, 1]),
    integer1=neps.Integer(lower=0, upper=1),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/working_directory_per_pipeline",
    max_evaluations_total=5,
)
