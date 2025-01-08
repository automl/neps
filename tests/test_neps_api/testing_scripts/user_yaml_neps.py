from __future__ import annotations

import logging
from pathlib import Path
from warnings import warn

import neps

pipeline_space = {
    "val1": neps.Float(lower=-10, upper=10),
    "val2": neps.Integer(lower=1, upper=5),
}


def run_pipeline(val1, val2):
    warn(
        "run_pipeline is deprecated, use evaluate_pipeline instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return evaluate_pipeline(val1, val2)


def evaluate_pipeline(val1, val2):
    return val1 * val2


logging.basicConfig(level=logging.INFO)

# Testing using created yaml with api
script_directory = Path(__file__).resolve().parent
parent_directory = script_directory.parent
searcher_path = Path(parent_directory) / "testing_yaml" / "optimizer_test"
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="user_yaml_bo",
    max_evaluations_total=1,
    searcher=searcher_path,
    initial_design_size=5,
)
