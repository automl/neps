from neps.utils.run_args import Settings, Default
import pytest
import neps
from neps.utils.run_args import get_run_args_from_yaml
from tests.test_yaml_run_args.test_yaml_run_args import (
    run_pipeline,
    hook1,
    hook2,
    pipeline_space,
)
from neps.optimizers.bayesian_optimization.optimizer import BayesianOptimization
from typing import Union, Callable, Dict, List, Type

BASE_PATH = "tests/test_settings"
run_pipeline = run_pipeline
hook1 = hook1
hook2 = hook2
pipeline_space = pipeline_space
my_bayesian = BayesianOptimization


@pytest.mark.neps_api
@pytest.mark.parametrize(
    "func_args, yaml_args, expected_output",
    [
        (
            {  # only essential arguments provided by func_args, no yaml
                "run_pipeline": run_pipeline,
                "root_directory": "path/to/root_directory",
                "pipeline_space": pipeline_space,
                "run_args": Default(None),
                "overwrite_working_directory": Default(False),
                "post_run_summary": Default(True),
                "development_stage_id": Default(None),
                "task_id": Default(None),
                "max_evaluations_total": 10,
                "max_evaluations_per_run": Default(None),
                "continue_until_max_evaluation_completed": Default(False),
                "max_cost_total": Default(None),
                "ignore_errors": Default(False),
                "loss_value_on_error": Default(None),
                "cost_value_on_error": Default(None),
                "pre_load_hooks": Default(None),
                "searcher": Default("default"),
                "searcher_kwargs": {},
            },
            Default(None),
            {
                "run_pipeline": run_pipeline,
                "root_directory": "path/to/root_directory",
                "pipeline_space": pipeline_space,
                "overwrite_working_directory": False,
                "post_run_summary": True,
                "development_stage_id": None,
                "task_id": None,
                "max_evaluations_total": 10,
                "max_evaluations_per_run": None,
                "continue_until_max_evaluation_completed": False,
                "max_cost_total": None,
                "ignore_errors": False,
                "loss_value_on_error": None,
                "cost_value_on_error": None,
                "pre_load_hooks": None,
                "searcher": "default",
                "searcher_kwargs": {},
            },
        ),
        (
            {  # only required elements of run_args
                "run_pipeline": Default(None),
                "root_directory": Default(None),
                "pipeline_space": Default(None),
                "run_args": Default(None),
                "overwrite_working_directory": Default(False),
                "post_run_summary": Default(True),
                "development_stage_id": Default(None),
                "task_id": Default(None),
                "max_evaluations_total": Default(None),
                "max_evaluations_per_run": Default(None),
                "continue_until_max_evaluation_completed": Default(False),
                "max_cost_total": Default(None),
                "ignore_errors": Default(False),
                "loss_value_on_error": Default(None),
                "cost_value_on_error": Default(None),
                "pre_load_hooks": Default(None),
                "searcher": Default("default"),
                "searcher_kwargs": {},
            },
            "/run_args_required.yaml",
            {
                "run_pipeline": run_pipeline,
                "root_directory": "path/to/root_directory",
                "pipeline_space": pipeline_space,
                "overwrite_working_directory": False,
                "post_run_summary": True,
                "development_stage_id": None,
                "task_id": None,
                "max_evaluations_total": 10,
                "max_evaluations_per_run": None,
                "continue_until_max_evaluation_completed": False,
                "max_cost_total": None,
                "ignore_errors": False,
                "loss_value_on_error": None,
                "cost_value_on_error": None,
                "pre_load_hooks": None,
                "searcher": "default",
                "searcher_kwargs": {},
            },
        ),
        (
            {  # required via func_args, optional via yaml
                "run_pipeline": run_pipeline,
                "root_directory": "path/to/root_directory",
                "pipeline_space": pipeline_space,
                "run_args": "tests/path/to/run_args",  # will be ignored by Settings
                "overwrite_working_directory": Default(False),
                "post_run_summary": Default(True),
                "development_stage_id": Default(None),
                "task_id": Default(None),
                "max_evaluations_total": 10,
                "max_evaluations_per_run": Default(None),
                "continue_until_max_evaluation_completed": Default(False),
                "max_cost_total": Default(None),
                "ignore_errors": Default(False),
                "loss_value_on_error": Default(None),
                "cost_value_on_error": Default(None),
                "pre_load_hooks": Default(None),
                "searcher": Default("default"),
                "searcher_kwargs": {},
            },
            "/run_args_optional.yaml",
            {
                "run_pipeline": run_pipeline,
                "root_directory": "path/to/root_directory",
                "pipeline_space": pipeline_space,
                "overwrite_working_directory": True,
                "post_run_summary": False,
                "development_stage_id": None,
                "task_id": None,
                "max_evaluations_total": 10,
                "max_evaluations_per_run": None,
                "continue_until_max_evaluation_completed": False,
                "max_cost_total": None,
                "ignore_errors": False,
                "loss_value_on_error": None,
                "cost_value_on_error": None,
                "pre_load_hooks": None,
                "searcher": "hyperband",
                "searcher_kwargs": {},
            },
        ),
        (
            {  # overwrite all yaml values
                "run_pipeline": run_pipeline,
                "root_directory": "path/to/root_directory",
                "pipeline_space": pipeline_space,
                "run_args": "test",
                "overwrite_working_directory": False,
                "post_run_summary": True,
                "development_stage_id": 5,
                "task_id": None,
                "max_evaluations_total": 17,
                "max_evaluations_per_run": None,
                "continue_until_max_evaluation_completed": False,
                "max_cost_total": None,
                "ignore_errors": False,
                "loss_value_on_error": None,
                "cost_value_on_error": None,
                "pre_load_hooks": None,
                "searcher": "default",
                "searcher_kwargs": {},
            },
            "/overwrite_run_args.yaml",
            {
                "run_pipeline": run_pipeline,
                "root_directory": "path/to/root_directory",
                "pipeline_space": pipeline_space,
                "overwrite_working_directory": False,
                "post_run_summary": True,
                "development_stage_id": 5,
                "task_id": None,
                "max_evaluations_total": 17,
                "max_evaluations_per_run": None,
                "continue_until_max_evaluation_completed": False,
                "max_cost_total": None,
                "ignore_errors": False,
                "loss_value_on_error": None,
                "cost_value_on_error": None,
                "pre_load_hooks": None,
                "searcher": "default",
                "searcher_kwargs": {},
            },
        ),
        (
            {  # optimizer args special case
                "run_pipeline": run_pipeline,
                "root_directory": "path/to/root_directory",
                "pipeline_space": pipeline_space,
                "run_args": "test",
                "overwrite_working_directory": False,
                "post_run_summary": True,
                "development_stage_id": 5,
                "task_id": None,
                "max_evaluations_total": 17,
                "max_evaluations_per_run": None,
                "continue_until_max_evaluation_completed": False,
                "max_cost_total": None,
                "ignore_errors": False,
                "loss_value_on_error": None,
                "cost_value_on_error": None,
                "pre_load_hooks": None,
                "searcher": Default("default"),
                "searcher_kwargs": {
                    "initial_design_type": "max_budget",
                    "use_priors": False,
                    "random_interleave_prob": 0.0,
                    "sample_default_first": False,
                    "sample_default_at_target": False,
                },
            },
            "/run_args_optimizer_settings.yaml",
            {
                "run_pipeline": run_pipeline,
                "root_directory": "path/to/root_directory",
                "pipeline_space": pipeline_space,
                "overwrite_working_directory": False,
                "post_run_summary": True,
                "development_stage_id": 5,
                "task_id": None,
                "max_evaluations_total": 17,
                "max_evaluations_per_run": None,
                "continue_until_max_evaluation_completed": False,
                "max_cost_total": None,
                "ignore_errors": False,
                "loss_value_on_error": None,
                "cost_value_on_error": None,
                "pre_load_hooks": None,
                "searcher": {
                    "strategy": "hyperband",
                    "eta": 3,
                    "initial_design_type": "max_budget",
                    "use_priors": False,
                    "random_interleave_prob": 0.0,
                    "sample_default_first": False,
                    "sample_default_at_target": False,
                },
                "searcher_kwargs": {
                    "initial_design_type": "max_budget",
                    "use_priors": False,
                    "random_interleave_prob": 0.0,
                    "sample_default_first": False,
                    "sample_default_at_target": False,
                },
            },
        ),
        (
            {  # load optimizer with args
                "run_pipeline": Default(None),
                "root_directory": Default(None),
                "pipeline_space": Default(None),
                "run_args": Default(None),
                "overwrite_working_directory": Default(False),
                "post_run_summary": Default(True),
                "development_stage_id": Default(None),
                "task_id": Default(None),
                "max_evaluations_total": Default(None),
                "max_evaluations_per_run": Default(None),
                "continue_until_max_evaluation_completed": Default(False),
                "max_cost_total": Default(None),
                "ignore_errors": Default(False),
                "loss_value_on_error": Default(None),
                "cost_value_on_error": Default(None),
                "pre_load_hooks": Default(None),
                "searcher": Default("default"),
                "searcher_kwargs": {
                    "random_interleave_prob": 0.2,
                    "initial_design_size": 9,
                },
            },
            "/run_args_optimizer_outside.yaml",
            {
                "run_pipeline": run_pipeline,
                "root_directory": "path/to/root_directory",
                "pipeline_space": pipeline_space,
                "overwrite_working_directory": True,
                "post_run_summary": True,
                "development_stage_id": None,
                "task_id": None,
                "max_evaluations_total": 10,
                "max_evaluations_per_run": None,
                "continue_until_max_evaluation_completed": False,
                "max_cost_total": None,
                "ignore_errors": False,
                "loss_value_on_error": None,
                "cost_value_on_error": None,
                "pre_load_hooks": None,
                "searcher": my_bayesian,
                "searcher_kwargs": {
                    "acquisition": "EI",
                    "acquisition_sampler": "random",
                    "random_interleave_prob": 0.2,
                    "initial_design_size": 9,
                },
            },
        ),
    ],
)
def test_check_settings(func_args: Dict, yaml_args: str, expected_output: Dict) -> None:
    """
    Check if expected settings are set
    """
    if not isinstance(yaml_args, Default):
        yaml_args = BASE_PATH + yaml_args
    settings = Settings(func_args, yaml_args)
    print(settings)
    for key, value in expected_output.items():
        assert getattr(settings, key) == value


@pytest.mark.neps_api
@pytest.mark.parametrize(
    "func_args, yaml_args, error",
    [
        (
            {
                "root_directory": Default(None),
                "pipeline_space": Default(None),
                "run_args": Default(None),
                "overwrite_working_directory": Default(False),
                "post_run_summary": Default(True),
                "development_stage_id": Default(None),
                "task_id": Default(None),
                "max_evaluations_total": Default(None),
                "max_evaluations_per_run": Default(None),
                "continue_until_max_evaluation_completed": Default(False),
                "max_cost_total": Default(None),
                "ignore_errors": Default(False),
                "loss_value_on_error": Default(None),
                "cost_value_on_error": Default(None),
                "pre_load_hooks": Default(None),
                "searcher": Default("default"),
                "searcher_kwargs": {},
            },
            Default(None),
            ValueError,
        )
    ],
)
def test_settings_initialization_error(
    func_args: Dict, yaml_args: Union[str, Default], error: Exception
) -> None:
    """
    Test if Settings raises Error when essential arguments are missing
    """
    with pytest.raises(error):
        Settings(func_args, yaml_args)
