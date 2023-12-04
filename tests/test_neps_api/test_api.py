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


testing_scripts = [
    "default_neps",
    "baseoptimizer_neps",
    "user_yaml_neps",
]

examples_folder = Path(__file__, "..", "testing_scripts").resolve()
solution_folder = Path(__file__, "..", "solution_yamls").resolve()
neps_api_example_script = [
    examples_folder / f"{example}.py" for example in testing_scripts
]


@pytest.mark.neps_api
def test_default_examples(tmp_path):
    # Running the example files holding multiple neps.run commands.

    runpy.run_path(
        neps_api_example_script[0],
        run_name="__main__",
    )

    # Testing each folder with its corresponding expected dictionary
    for folder_name in os.listdir(tmp_path):
        folder_path = os.path.join(tmp_path, folder_name)

        assert os.path.exists(folder_path), f"Directory does not exist: {folder_path}"

        info_yaml_path = os.path.join(folder_path, ".optimizer_info.yaml")

        assert os.path.exists(
            str(info_yaml_path)
        ), f"File does not exist: {info_yaml_path}"

        # Load the YAML file
        with open(str(info_yaml_path)) as yaml_config:
            loaded_data = yaml.safe_load(yaml_config)

        with open(str(solution_folder / (folder_name + ".yaml"))) as solution_yaml:
            expected_data = yaml.safe_load(solution_yaml)

        assert loaded_data == expected_data


@pytest.mark.neps_api
def test_baseoptimizer_examples(tmp_path):
    # Running the example files holding multiple neps.run commands.

    runpy.run_path(
        neps_api_example_script[1],
        run_name="__main__",
    )

    # Testing each folder with its corresponding expected dictionary
    for folder_name in os.listdir(tmp_path):
        folder_path = os.path.join(tmp_path, folder_name)

        assert os.path.exists(folder_path), f"Directory does not exist: {folder_path}"

        info_yaml_path = os.path.join(folder_path, ".optimizer_info.yaml")

        assert os.path.exists(
            str(info_yaml_path)
        ), f"File does not exist: {info_yaml_path}"

        # Load the YAML file
        with open(str(info_yaml_path)) as yaml_config:
            loaded_data = yaml.safe_load(yaml_config)

        with open(str(solution_folder / (folder_name + ".yaml"))) as solution_yaml:
            expected_data = yaml.safe_load(solution_yaml)

        assert loaded_data == expected_data


@pytest.mark.neps_api
def test_user_created_yaml_examples(tmp_path):
    runpy.run_path(
        neps_api_example_script[2],
        run_name="__main__",
    )

    # Testing each folder with its corresponding expected dictionary
    for folder_name in os.listdir(tmp_path):
        folder_path = os.path.join(tmp_path, folder_name)

        assert os.path.exists(folder_path), f"Directory does not exist: {folder_path}"

        info_yaml_path = os.path.join(folder_path, ".optimizer_info.yaml")

        assert os.path.exists(
            str(info_yaml_path)
        ), f"File does not exist: {info_yaml_path}"

        # Load the YAML file
        with open(str(info_yaml_path)) as yaml_config:
            loaded_data = yaml.safe_load(yaml_config)

        with open(str(solution_folder / (folder_name + ".yaml"))) as solution_yaml:
            expected_data = yaml.safe_load(solution_yaml)

        assert loaded_data == expected_data
