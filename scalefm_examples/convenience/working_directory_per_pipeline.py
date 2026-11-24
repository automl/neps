import logging
from pathlib import Path
from warnings import warn

import numpy as np

import scalefm


def evaluate_pipeline(pipeline_directory: Path, float1, categorical, integer1):
    # When adding pipeline_directory to evaluate_pipeline, scalefm detects its presence and
    # passes a directory unique for each pipeline configuration. You can then use this
    # pipeline_directory to create / save files pertaining to a specific pipeline, e.g.:
    pipeline_info = pipeline_directory / "info_file.txt"
    pipeline_info.write_text(f"{float1} - {categorical} - {integer1}")

    objective_to_minimize = -float(np.sum([float1, int(categorical), integer1]))
    return {
        "objective_to_minimize": objective_to_minimize,
        "cost": 1.0,  # The training cost of this run, e.g. its FLOPs
    }


class HPOSpace(scalefm.PipelineSpace):
    float1 = scalefm.Float(lower=0, upper=1)
    categorical = scalefm.Categorical(choices=(0, 1))
    integer1 = scalefm.Integer(lower=0, upper=1)


logging.basicConfig(level=logging.INFO)
scalefm.acquire(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=HPOSpace(),
    root_directory="results/working_directory_per_pipeline",
    worker_evaluations_to_spend=5,
)
