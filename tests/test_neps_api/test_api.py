import logging
import os
import runpy
from pathlib import Path

import pytest
import yaml


# To change the working directly into the tmp_path when testing function
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


HERE = Path(__file__).resolve().parent

testing_scripts = ["default_neps", "baseoptimizer_neps", "user_yaml_neps"]
EXAMPLES_FOLDER = HERE / "testing_scripts"
SOLUTION_FOLDER = HERE / "solution_yamls"
neps_api_example_script = [
    EXAMPLES_FOLDER / f"{example}.py" for example in testing_scripts
]


@pytest.mark.neps_api
@pytest.mark.parametrize("example_script", neps_api_example_script)
def test_default_examples(tmp_path: Path, example_script: Path) -> None:
    # Running the example files holding multiple neps.run commands.
    runpy.run_path(str(example_script), run_name="__main__")

    # Testing each folder with its corresponding expected dictionary
    for folder in tmp_path.iterdir():
        info_yaml_path = folder / "optimizer_info.yaml"

        assert info_yaml_path.exists()
        loaded_data = yaml.safe_load(info_yaml_path.read_text())

        solution_yaml_path = SOLUTION_FOLDER / (folder.name + ".yaml")
        solution_data = yaml.safe_load(solution_yaml_path.read_text())

        assert (
            loaded_data == solution_data
        ), f"Solution Path: {solution_yaml_path}\nLoaded Path: {info_yaml_path}\n"
