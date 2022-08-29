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
    "expert_priors/architecture_and_hyperparameters",
    "multi_fidelity/optimize",
    "experimental/cost_aware",
]
all_examples = core_examples + [  # Run on github actions
    "basic_usage/architecture_and_hyperparameters",
    "basic_usage/hierarchical_architecture",
    "expert_priors/hyperparameters",
    "experimental/hierarchical_architecture_hierarchical_GP",
]


examples_folder = Path(__file__, "..", "..", "neps_examples").resolve()
core_examples_scripts = [examples_folder / f"{example}.py" for example in core_examples]
all_examples_scripts = [examples_folder / f"{example}.py" for example in all_examples]


@pytest.mark.all_examples
@pytest.mark.parametrize("example", all_examples_scripts, ids=all_examples)
def test_all_examples(example):
    runpy.run_path(example, run_name="__main__")


@pytest.mark.core_examples
@pytest.mark.parametrize("example", core_examples_scripts, ids=core_examples)
def test_core_examples(example):
    runpy.run_path(example, run_name="__main__")
