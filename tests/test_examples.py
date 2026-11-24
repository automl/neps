from __future__ import annotations

import logging
import os
import runpy
from pathlib import Path

import pytest
from scalefm_examples import ci_examples


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


examples_folder = Path(__file__, "..", "..", "scalefm_examples").resolve()
ci_examples_scripts = [examples_folder / f"{example}.py" for example in ci_examples]


@pytest.mark.ci_examples
@pytest.mark.parametrize("example", ci_examples_scripts, ids=ci_examples)
def test_ci_examples(example):
    runpy.run_path(str(example), run_name="__main__")
