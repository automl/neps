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


# Expected outcomes of the optimizer YAML according to different cases of neps.run
# all based on the examples in tests/test_neps_api/examples_test_api.py
expected_dicts = {
    "priorband_bo_user_decided": {
        "searcher_name": "priorband_bo",
        "searcher_alg": "priorband",
        "user_defined_searcher": True,
        "args_accepted_changes": True,
    },
    "priorband_neps_decided": {
        "searcher_name": "priorband",
        "searcher_alg": "priorband",
        "user_defined_searcher": False,
        "args_accepted_changes": False,
    },
    "bo_neps_decided": {
        "searcher_name": "bayesian_optimization",
        "searcher_alg": "bayesian_optimization",
        "user_defined_searcher": False,
        "args_accepted_changes": None,
    },
    "pibo_neps_decided": {
        "searcher_name": "pibo",
        "searcher_alg": "bayesian_optimization",
        "user_defined_searcher": False,
        "args_accepted_changes": False,
    },
    "hyperband_neps_decided": {
        "searcher_name": "hyperband",
        "searcher_alg": "hyperband",
        "user_defined_searcher": False,
        "args_accepted_changes": False,
    },
}


yaml_api_example = "examples_test_api"  # Run locally and on github actions

examples_folder = Path(__file__, "..").resolve()
yaml_api_example_script = examples_folder / f"{yaml_api_example}.py"


@pytest.mark.yaml_api
def test_core_examples(tmp_path):
    # Running the example files holding multiple neps.run commands.
    runpy.run_path(yaml_api_example_script, run_name="__main__")

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

        assert loaded_data == expected_dicts[folder_name]
