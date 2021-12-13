import os
import runpy
from pathlib import Path

import pytest

# Collect python scripts in the examples folder
exclude_tests = {"hyperparameters_architecture"}

examples_folder = Path(__file__, "..", "..", "neps_examples").resolve()
example_files = [
    example_folder / "optimize.py" for example_folder in examples_folder.iterdir()
]
example_files = [
    example_file
    for example_file in example_files
    if example_file.exists() and example_file.parent.name not in exclude_tests
]
example_files_names = [example_file.parent.name for example_file in example_files]


@pytest.fixture(autouse=True)
def use_tmpdir(tmp_path, request):
    os.chdir(tmp_path)
    yield
    os.chdir(request.config.invocation_dir)


@pytest.mark.parametrize("example", example_files, ids=example_files_names)
def test_examples(example):
    runpy.run_path(example, run_name="__main__")
