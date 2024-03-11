import pytest
from neps.utils.run_args_from_yaml import get_run_args_from_yaml

BASE_PATH = "tests/test_yaml_run_args/"


def run_pipeline():
    """func to test loading of run_pipeline"""
    return


def hook1():
    """func to test loading of pre_load_hooks"""
    return


def hook2():
    """func to test loading of pre_load_hooks"""
    return


def check_run_args(yaml_path_run_args, expected_output):
    """
    Validates the loaded NEPS configuration against expected settings.

    Loads NEPS configuration settings from a specified YAML file and verifies
    against expected settings, including function objects. Special handling is
    applied to compare functions.

    Args:
        yaml_path_run_args (str): The relative path to the YAML configuration file.
        expected_output (dict): The expected NEPS configuration settings.

    Raises:
        AssertionError: If any configuration setting does not match the expected value.
    """
    output = get_run_args_from_yaml(BASE_PATH + yaml_path_run_args)

    def are_functions_equivalent(f1, f2):
        """
        Compares two functions or lists of functions for equivalence based on their
        bytecode.

        Determines if the provided functions or each function within the provided lists
        have identical bytecode, implying they perform the same operations. This approach
        is useful for comparing two different instances to assess if they originate from
        the same root function.

        Args:
            f1: A function or list of functions.
            f2: Another function or list of functions to compare against f1.

        Returns:
            bool: True if the functions or all functions in the lists are equivalent,
            False otherwise.
        """
        if isinstance(f1, list) and isinstance(f2, list):
            if len(f1) != len(f2):
                return False
            return all(
                f1_item.__code__.co_code == f2_item.__code__.co_code
                for f1_item, f2_item in zip(f1, f2)
            )
        return f1.__code__.co_code == f2.__code__.co_code

    # Remove and compare special function keys separately
    # Special because they include a module loading procedure by a path and the name of
    # the function
    for special_key in ["run_pipeline", "pre_load_hooks"]:
        if special_key in expected_output:
            func_expected = expected_output.pop(special_key)
            func_output = output.pop(special_key)
            assert are_functions_equivalent(func_expected, func_output), (
                f"Mismatch in {special_key} " f"function(s)"
            )

    # Assert that the output matches the expected output
    assert output == expected_output, f"Expected {expected_output}, but got {output}"


@pytest.mark.neps_api
@pytest.mark.parametrize(
    "yaml_path,expected_output",
    [
        (
            "run_args_full.yaml",
            {
                "run_pipeline": run_pipeline,
                "pipeline_space": "pipeline_space.yaml",
                "root_directory": "test_yaml",
                "max_evaluations_total": 20,
                "max_cost_total": 3,
                "overwrite_working_directory": True,
                "post_run_summary": True,
                "development_stage_id": "Early_Stage",
                "task_id": 4,
                "max_evaluations_per_run": 5,
                "continue_until_max_evaluation_completed": True,
                "loss_value_on_error": 4.2,
                "cost_value_on_error": 3.7,
                "ignore_errors": True,
                "searcher": "bayesian_optimization",
                "searcher_path": "/path/to/model",
                "searcher_kwargs": {"initial_design_size": 5, "surrogate_model": "gp"},
                "pre_load_hooks": [hook1, hook2],
            },
        ),
        (
            "run_args_full_same_level.yaml",
            {
                "run_pipeline": run_pipeline,
                "pipeline_space": "pipeline_space.yaml",
                "root_directory": "test_yaml",
                "max_evaluations_total": 20,
                "max_cost_total": 4.2,
                "overwrite_working_directory": True,
                "post_run_summary": False,
                "development_stage_id": 9,
                "task_id": 2.0,
                "max_evaluations_per_run": 5,
                "continue_until_max_evaluation_completed": True,
                "loss_value_on_error": 2.4,
                "cost_value_on_error": 2.1,
                "ignore_errors": False,
                "searcher": "bayesian_optimization",
                "searcher_path": "/path/to/searcher",
                "searcher_kwargs": {"initial_design_size": 5, "surrogate_model": "gp"},
                "pre_load_hooks": [hook1],
            },
        ),
        (
            "run_args_partial.yaml",
            {
                "pipeline_space": "pipeline_space.yaml",
                "root_directory": "test_yaml",
                "max_evaluations_total": 20,
                "overwrite_working_directory": True,
                "post_run_summary": False,
                "continue_until_max_evaluation_completed": False,
                "searcher": "bayesian_optimization",
                "searcher_kwargs": {"initial_design_size": 5, "surrogate_model": "gp"},
            },
        ),
        (
            "run_args_partial_same_level.yaml",
            {
                "root_directory": "test_yaml",
                "max_evaluations_total": 20,
                "overwrite_working_directory": True,
                "post_run_summary": False,
                "task_id": 4,
                "continue_until_max_evaluation_completed": True,
                "ignore_errors": True,
            },
        ),
        ("run_args_empty.yaml", {}),
    ],
)
def test_yaml_config(yaml_path, expected_output):
    """
    Parameterized test for verifying NEPS configuration loading from YAML.

    Each test case supplies a YAML file path and expected configuration settings,
    assessing the accuracy of the configuration loading functionality across various
    scenarios. This includes tests for different levels of hierarchy in configuration
    settings, as well as comparisons between partial and full definitions.

    Args:
        yaml_path (str): Path to the YAML file being tested.
        expected_output (dict): Dictionary of the expected configuration settings.
    """
    check_run_args(yaml_path, expected_output)


@pytest.mark.neps_api
@pytest.mark.parametrize(
    "yaml_path, expected_exception",
    [
        ("run_args_invalid_type.yaml", TypeError),
        ("run_args_wrong_path.yaml", ImportError),
        ("run_args_invalid_key.yaml", KeyError),
        ("run_args_wrong_name.yaml", ImportError),
        ("run_args_key_missing.yaml", KeyError),
    ],
)
def test_yaml_failure_cases(yaml_path, expected_exception):
    """
    Tests various error scenarios for loading NEPS configuration from YAML files.

    This parameterized test function verifies that the `get_run_args_from_yaml` function
    correctly raises the expected exceptions for different types of configuration errors.
    Each test case simulates a common error scenario, including invalid types, missing
    keys, and incorrect paths or names within the YAML files.

    Args:
        yaml_path (str): The path to the YAML file containing the erroneous configuration.
        expected_exception (Exception): The type of exception expected to be raised.

    The test ensures robust error handling within the configuration loading process.
    """
    with pytest.raises(expected_exception):
        get_run_args_from_yaml(BASE_PATH + yaml_path)
