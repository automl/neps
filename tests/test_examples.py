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


# Collect python scripts in the examples folder
disabled_examples = {"fault_tolerance"}
examples_folder = Path(__file__, "..", "..", "neps_examples").resolve()
example_files = [
    example_folder / "optimize.py"
    for example_folder in examples_folder.iterdir()
    if example_folder.name not in disabled_examples
]
example_files = [example_file for example_file in example_files if example_file.exists()]
example_files_names = [example_file.parent.name for example_file in example_files]


@pytest.mark.all_examples
@pytest.mark.parametrize("example", example_files, ids=example_files_names)
def test_all_examples(example):
    runpy.run_path(example, run_name="__main__")


core_example_names = [
    "user_priors_also_architecture",
    "cost_aware",
    "hyperparameters",
    "multi_fidelity",
]
core_example_files = [
    example_file
    for example_file in example_files
    if example_file.parent.name in core_example_names
]


@pytest.mark.core_examples
@pytest.mark.parametrize("example", core_example_files, ids=core_example_names)
def test_core_examples(example):
    runpy.run_path(example, run_name="__main__")
