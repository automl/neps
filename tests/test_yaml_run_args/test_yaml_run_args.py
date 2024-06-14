import pytest
import neps
from neps.utils.run_args import get_run_args_from_yaml
from neps.optimizers.bayesian_optimization.optimizer import BayesianOptimization
from typing import Union, Callable, Dict, List, Type

BASE_PATH = "tests/test_yaml_run_args/"
pipeline_space = dict(lr=neps.FloatParameter(lower=1e-3, upper=0.1),
                      optimizer=neps.CategoricalParameter(choices=["adam", "sgd",
                                                                   "adamw"]),
                      epochs=neps.IntegerParameter(lower=1, upper=10),
                      batch_size=neps.ConstantParameter(value=64))


def run_pipeline():
    """func to test loading of run_pipeline"""
    return


def hook1():
    """func to test loading of pre_load_hooks"""
    return


def hook2():
    """func to test loading of pre_load_hooks"""
    return


def check_run_args(yaml_path_run_args: str, expected_output: Dict) -> None:
    """
    Validates the loaded NEPS configuration against expected settings.

    Loads NEPS configuration settings from a specified YAML file and verifies
    against expected settings, including function objects, dict and classes. Special
    handling is applied to compare functions.

    Args:
        yaml_path_run_args (str): The path to the YAML configuration file.
        expected_output (dict): The expected NePS configuration settings.

    Raises:
        AssertionError: If any configuration setting does not match the expected value.
    """
    output = get_run_args_from_yaml(BASE_PATH + yaml_path_run_args)

    def are_functions_equivalent(f1: Union[Callable, List[Callable]],
                                 f2: Union[Callable, List[Callable]]) -> bool:
        """
        Compares functions or lists of functions for equivalence by their bytecode,
        useful when identical functions have different memory addresses. This method
        identifies if functions, despite being distinct instances, perform identical
        operations.

        Parameters:
        - func1: Function or list of functions to compare.
        - func2: Function or list of functions to compare against func1.

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

    # Compare keys with a function/list of functions as their values
    # Special because they include a module loading procedure by a path and the name of
    # the function
    for special_key in ["run_pipeline", "pre_load_hooks"]:
        if special_key in expected_output:
            func_expected = expected_output.pop(special_key)
            func_output = output.pop(special_key)
            assert are_functions_equivalent(func_expected, func_output), (
                f"Mismatch in {special_key} " f"function(s)"
            )
    # Compare instances of a subclass of BaseOptimizer
    if "searcher" in expected_output and not isinstance(expected_output["searcher"], str):
        # 'searcher': BaseOptimizer()
        optimizer_expected = expected_output.pop("searcher")
        optimizer_output = output.pop("searcher", None)
        assert isinstance(optimizer_output, type(optimizer_expected))

    # Assert that the rest of the output dict matches the expected output dict
    assert output == expected_output, f"Expected {expected_output}, but got {output}"


@pytest.mark.neps_api
@pytest.mark.parametrize(
    "yaml_path,expected_output",
    [
        (
            "run_args_full.yaml",
            {
                "run_pipeline": run_pipeline,
                "pipeline_space": pipeline_space,
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
                "searcher_kwargs": {"initial_design_size": 5, "surrogate_model": "gp"},
                "pre_load_hooks": [hook1, hook2],
            },
        ),
        (
            "run_args_full_same_level.yaml",
            {
                "run_pipeline": run_pipeline,
                "pipeline_space": pipeline_space,
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
                "searcher_kwargs": {"initial_design_size": 5, "surrogate_model": "gp"},
                "pre_load_hooks": [hook1],
            },
        ),
        (
            "run_args_partial.yaml",
            {
                "pipeline_space": pipeline_space,
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
        ("run_args_optional_loading_format.yaml", {
            "run_pipeline": run_pipeline,
            "pipeline_space": pipeline_space,
            "root_directory": "test_yaml",
            "max_evaluations_total": 20,
            "max_cost_total": 4.2,
            "overwrite_working_directory": True,
            "post_run_summary": False,
            "development_stage_id": 9,
            "max_evaluations_per_run": 5,
            "continue_until_max_evaluation_completed": True,
            "loss_value_on_error": 2.4,
            "cost_value_on_error": 2.1,
            "ignore_errors": False,
            "searcher": BayesianOptimization,
            "searcher_kwargs": {
                "initial_design_size": 5,
                "surrogate_model": "gp"
            },
            "pre_load_hooks": [hook1]

        })
    ],
)
def test_yaml_config(yaml_path: str, expected_output: Dict) -> None:
    """
    Tests NePS configuration loading from run_args=YAML, comparing expected settings
    against loaded ones. Covers hierarchical levels and partial/full of yaml
    dict definitions.

    Args:
        yaml_path (str): Path to the YAML file.
        expected_output (dict): Expected configuration settings.
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
def test_yaml_failure_cases(yaml_path: str, expected_exception: Type[Exception]) -> None:
    """
    Tests for expected exceptions when loading erroneous NePS configurations from YAML.

    Each case checks if `get_run_args_from_yaml` raises the correct exception for errors
    like invalid types, missing keys, and incorrect paths in YAML configurations.

    Args:
        yaml_path (str): Path to the error-containing YAML file.
        expected_exception (Exception): Expected exception type.
    """
    with pytest.raises(expected_exception):
        get_run_args_from_yaml(BASE_PATH + yaml_path)
