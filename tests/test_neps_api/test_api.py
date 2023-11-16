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
        record
        for record in caplog.get_records("call")
        if record.levelno >= logging.ERROR
    ]
    assert not errors


# Expected outcomes of the optimizer YAML according to different cases of neps.run
# all based on the examples in tests/test_neps_api/examples_test_api.py
expected_dicts = {
    "priorband_bo_user_decided": {
        "searcher_name": "priorband_bo",
        "searcher_alg": "priorband",
        "searcher_selection_source": "Default_Searcher-User_Choice",
        "searcher_modified_arguments": {
            "initial_design_size": 5,
        },
    },
    "bo_user_decided": {
        "searcher_name": "bayesian_optimization",
        "searcher_alg": "bayesian_optimization",
        "searcher_selection_source": "Default_Searcher-User_Choice",
        "searcher_modified_arguments": {
            "surrogate_model": "ComprehensiveGPHierarchy",
            "surrogate_model_args": {
                "graph_kernels": [
                    "WeisfeilerLehman",
                    "WeisfeilerLehman",
                    "WeisfeilerLehman",
                    "WeisfeilerLehman",
                    "WeisfeilerLehman",
                ],
                "hp_kernels": [],
                "verbose": False,
                "hierarchy_consider": [0, 1, 2, 3],
                "d_graph_features": 0,
                "vectorial_features": None,
            },
        },
    },
    "priorband_neps_decided": {
        "searcher_name": "priorband",
        "searcher_alg": "priorband",
        "searcher_selection_source": "Default_Searcher-NePS_Decision_Tree",
        "searcher_modified_arguments": {},
    },
    "bo_neps_decided": {
        "searcher_name": "bayesian_optimization",
        "searcher_alg": "bayesian_optimization",
        "searcher_selection_source": "Default_Searcher-NePS_Decision_Tree",
        "searcher_modified_arguments": {},
    },
    "pibo_neps_decided": {
        "searcher_name": "pibo",
        "searcher_alg": "bayesian_optimization",
        "searcher_selection_source": "Default_Searcher-NePS_Decision_Tree",
        "searcher_modified_arguments": {},
    },
    "hyperband_neps_decided": {
        "searcher_name": "hyperband",
        "searcher_alg": "hyperband",
        "searcher_selection_source": "Default_Searcher-NePS_Decision_Tree",
        "searcher_modified_arguments": {},
    },
    "bo_custom_created": {
        "searcher_name": "custom",
        "searcher_alg": "BayesianOptimization",
        "searcher_selection_source": "Custom-BaseOptimizer",
        "searcher_modified_arguments": {},
    },
    "hyperband_custom_created": {
        "searcher_name": "custom",
        "searcher_alg": "Hyperband",
        "searcher_selection_source": "Custom-BaseOptimizer",
        "searcher_modified_arguments": {},
    },
    "user_yaml_bo": {
        "searcher_name": "optimizer_test",
        "searcher_alg": "bayesian_optimization",
        "searcher_selection_source": "Custom-User_Yaml",
        "searcher_modified_arguments": {
            "initial_design_size": 5,
        },
    },
}


# Run locally and on github actions

testing_scripts = [
    "default_neps",
    "baseoptimizer_neps",
    "user_yaml_neps",
]

examples_folder = Path(__file__, "..", "testing_scripts").resolve()
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

        assert loaded_data == expected_dicts[folder_name]


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

        assert loaded_data == expected_dicts[folder_name]


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

        assert loaded_data == expected_dicts[folder_name]
