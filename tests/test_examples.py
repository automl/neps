import logging
import os
import runpy
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def use_tmpdir(tmp_path, request):
    os.chdir(tmp_path)
    yield
    os.chdir(request.config.invocation_dir)


# https://stackoverflow.com/a/59745629
# Fail tests if there is a logging.error
@pytest.fixture(autouse=True)
def no_logs_gte_error(caplog):
    yield
    errors = [
        record for record in caplog.get_records("call") if record.levelno >= logging.ERROR
    ]
    assert not errors


core_examples = [  # Run locally and on github actions
    "basic_usage/hyperparameters",
    "basic_usage/analyse",
    "experimental/expert_priors_for_architecture_and_hyperparameters",
    "efficiency/multi_fidelity",
]
all_examples = core_examples + [  # Run on github actions
    "basic_usage/architecture_and_hyperparameters",
    "experimental/hierarchical_architecture",
    "efficiency/expert_priors_for_hyperparameters",
    "experimental/hierarchical_architecture_hierarchical_GP",
    "convenience/logging_additional_info",
    "convenience/working_directory_per_pipeline",
]


examples_folder = Path(__file__, "..", "..", "neps_examples").resolve()
core_examples_scripts = [examples_folder / f"{example}.py" for example in core_examples]
all_examples_scripts = [examples_folder / f"{example}.py" for example in all_examples]


@pytest.mark.core_examples
@pytest.mark.parametrize("example", core_examples_scripts, ids=core_examples)
def test_core_examples(example):
    if example.name == "analyse.py":
        # Run hyperparameters example to have something to analyse
        runpy.run_path(core_examples_scripts[0], run_name="__main__")

    runpy.run_path(example, run_name="__main__")


@pytest.mark.all_examples
@pytest.mark.parametrize("example", all_examples_scripts, ids=all_examples)
def test_all_examples(example):
    test_core_examples(example)
