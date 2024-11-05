import logging
import os
from pathlib import Path
from warnings import warn
import neps

pipeline_space = dict(
    val1=neps.Float(lower=-10, upper=10),
    val2=neps.Integer(lower=1, upper=5),
)


def run_pipeline(val1, val2):
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning)
    loss = val1 * val2
    return loss


logging.basicConfig(level=logging.INFO)

# Testing using created yaml with api
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.join(script_directory, os.pardir)
searcher_path = Path(parent_directory) / "testing_yaml" / "optimizer_test"
neps.run(
    evaluate_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="user_yaml_bo",
    max_evaluations_total=1,
    searcher=searcher_path,
    initial_design_size=5,
)
