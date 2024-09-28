import logging
import os
from pathlib import Path
import neps

pipeline_space = dict(
    val1=neps.Float(lower=-10, upper=10),
    val2=neps.IntegerParameter(lower=1, upper=5),
)


def run_pipeline(val1, val2):
    loss = val1 * val2
    return loss


logging.basicConfig(level=logging.INFO)

# Testing using created yaml with api
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.join(script_directory, os.pardir)
searcher_path = Path(parent_directory) / "testing_yaml" / "optimizer_test"
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="user_yaml_bo",
    max_evaluations_total=1,
    searcher=searcher_path,
    initial_design_size=5,
)
