from __future__ import annotations

import logging
import os
import runpy
from pathlib import Path

import pytest
from neps_examples import ci_examples, core_examples


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


examples_folder = Path(__file__, "..", "..", "neps_examples").resolve()
core_examples_scripts = [examples_folder / f"{example}.py" for example in core_examples]
ci_examples_scripts = [examples_folder / f"{example}.py" for example in ci_examples]


@pytest.mark.core_examples
@pytest.mark.parametrize("example", core_examples_scripts, ids=core_examples)
def test_core_examples(example):
    if example.name == "analyse.py":
        # Run hyperparameters example to have something to analyse
        runpy.run_path(str(core_examples_scripts[0]), run_name="__main__")

    if example.name in (
        "architecture.py",
        "architecture_and_hyperparameters.py",
        "hierarchical_architecture.py",
        "expert_priors_for_architecture_and_hyperparameters.py",
    ):
        pytest.xfail("Architecture were removed temporarily")

    runpy.run_path(str(example), run_name="__main__")


@pytest.mark.ci_examples
@pytest.mark.parametrize("example", ci_examples_scripts, ids=ci_examples)
def test_ci_examples(example):
    test_core_examples(example)
