from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

from neps.optimizers.base_optimizer import BaseOptimizer
from neps.search_spaces.search_space import pipeline_space_from_yaml

logger = logging.getLogger("neps")

# Define the name of the arguments as variables for easier code maintenance
RUN_ARGS = "run_args"
RUN_PIPELINE = "run_pipeline"
PIPELINE_SPACE = "pipeline_space"
ROOT_DIRECTORY = "root_directory"
MAX_EVALUATIONS_TOTAL = "max_evaluations_total"
MAX_COST_TOTAL = "max_cost_total"
OVERWRITE_WORKING_DIRECTORY = "overwrite_working_directory"
POST_RUN_SUMMARY = "post_run_summary"
DEVELOPMENT_STAGE_ID = "development_stage_id"
TASK_ID = "task_id"
CONTINUE_UNTIL_MAX_EVALUATION_COMPLETED = "continue_until_max_evaluation_completed"
LOSS_VALUE_ON_ERROR = "loss_value_on_error"
COST_VALUE_ON_ERROR = "cost_value_on_error"
IGNORE_ERROR = "ignore_errors"
SEARCHER = "searcher"
PRE_LOAD_HOOKS = "pre_load_hooks"
SEARCHER_KWARGS = "searcher_kwargs"
MAX_EVALUATIONS_PER_RUN = "max_evaluations_per_run"


def get_run_args_from_yaml(path: str) -> dict:
    """Load and validate NEPS run arguments from a specified YAML configuration file
    provided via run_args.

    This function reads a YAML file, extracts the arguments required by NEPS,
    validates these arguments, and then returns them in a dictionary. It checks for the
    presence and validity of expected parameters, and distinctively handles more complex
    configurations, specifically those that are dictionaries(e.g. pipeline_space) or
    objects(e.g. run_pipeline) requiring loading.

    Parameters:
        path (str): The file path to the YAML configuration file.

    Returns:
        A dictionary of validated run arguments.

    Raises:
        KeyError: If any parameter name is invalid.
    """
    # Load the YAML configuration file
    config = config_loader(path)

    # Initialize an empty dictionary to hold the extracted settings
    settings = {}

    # List allowed NEPS run arguments with simple types (e.g., string, int). Parameters
    # like 'searcher_kwargs', 'run_pipeline', 'preload_hooks', 'pipeline_space',
    # and 'searcher' are excluded due to needing specialized processing.
    expected_parameters = [
        ROOT_DIRECTORY,
        MAX_EVALUATIONS_TOTAL,
        MAX_COST_TOTAL,
        OVERWRITE_WORKING_DIRECTORY,
        POST_RUN_SUMMARY,
        DEVELOPMENT_STAGE_ID,
        TASK_ID,
        MAX_EVALUATIONS_PER_RUN,
        CONTINUE_UNTIL_MAX_EVALUATION_COMPLETED,
        LOSS_VALUE_ON_ERROR,
        COST_VALUE_ON_ERROR,
        IGNORE_ERROR,
    ]

    # Flatten the YAML file's structure to separate flat parameters (flat_config) and
    # those needing special handling (special_configs).
    flat_config, special_configs = extract_leaf_keys(config)

    # Check if flatten dict (flat_config) just contains the expected parameters
    for parameter, value in flat_config.items():
        if parameter in expected_parameters:
            settings[parameter] = value
        else:
            raise KeyError(
                f"Parameter '{parameter}' is not an argument of neps.run() "
                f"provided via run_args."
                f"See here all valid arguments:"
                f" {', '.join(expected_parameters)}, "
                f"'run_pipeline', 'preload_hooks', 'pipeline_space'"
            )

    # Process complex configurations (e.g., 'pipeline_space', 'searcher') and integrate
    # them into 'settings'.
    handle_special_argument_cases(settings, special_configs)

    # check if all provided arguments have legal types
    check_run_args(settings)

    logger.debug(
        f"The 'run_args' arguments: {settings} are now extracted and type-tested from "
        f"referenced YAML."
    )

    return settings


def config_loader(path: str) -> dict:
    """Loads a YAML file and returns the contents under the 'run_args' key.

    Validates the existence and format of the YAML file and checks for the presence of
    the 'run_args' as the only top level key. If any conditions are not met,
    raises an
    exception with a helpful message.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Contents under the 'run_args' key.

    Raises:
        FileNotFoundError: If the file at 'path' does not exist.
        ValueError: If the file is not a valid YAML.
        KeyError: If 'run_args' key is missing.
        KeyError: If 'run_args' is not the only top level key
    """
    try:
        with open(path) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"The specified file was not found: '{path}'."
            f" Please make sure that the path is correct and "
            f"try again."
        ) from e
    except yaml.YAMLError as e:
        raise ValueError(f"The file at {path} is not a valid YAML file.") from e

    return config


def extract_leaf_keys(d: dict, special_keys: dict | None = None) -> tuple[dict, dict]:
    """Recursive function to extract leaf keys and their values from a nested dictionary.
    Special keys (e.g. 'searcher_kwargs', 'run_pipeline') are also extracted if present
    and their corresponding values (dict) at any level in the nested structure.

    :param d: The dictionary to extract values from.
    :param special_keys: A dictionary to store values of special keys.
    :return: A tuple containing the leaf keys dictionary and the dictionary for
    special keys.
    """
    if special_keys is None:
        special_keys = {
            SEARCHER_KWARGS: None,
            RUN_PIPELINE: None,
            PRE_LOAD_HOOKS: None,
            SEARCHER: None,
            PIPELINE_SPACE: None,
        }

    leaf_keys = {}
    for k, v in d.items():
        if k in special_keys and v != "None":
            special_keys[k] = v
        elif isinstance(v, dict):
            # Recursively call to explore nested dictionaries
            nested_leaf_keys, _ = extract_leaf_keys(v, special_keys)
            leaf_keys.update(nested_leaf_keys)
        elif v is not None and v != "None":
            leaf_keys[k] = v
    return leaf_keys, special_keys


def handle_special_argument_cases(settings: dict, special_configs: dict) -> None:
    """Process and integrate special configuration cases into the 'settings' dictionary.

    This function updates 'settings' with values from 'special_configs'. It handles
    specific keys that require more complex processing, such as 'pipeline_space' and
    'searcher', which may need to load a function/dict from paths. It also manages nested
    configurations like 'searcher_kwargs' and 'pre_load_hooks' which need individual
    processing or function loading.

    Parameters:
    - settings (dict): The dictionary to be updated with processed configurations.
    - special_configs (dict): A dictionary containing configuration keys and values
                              that require special processing.

    Returns:
    - None: The function modifies 'settings' in place.

    """
    # process special configs
    process_run_pipeline(RUN_PIPELINE, special_configs, settings)
    process_pipeline_space(PIPELINE_SPACE, special_configs, settings)
    process_searcher(SEARCHER, special_configs, settings)

    if special_configs[PRE_LOAD_HOOKS] is not None:
        # Loads the pre_load_hooks functions and add them in a list to settings.
        settings[PRE_LOAD_HOOKS] = load_hooks_from_config(special_configs[PRE_LOAD_HOOKS])


def process_pipeline_space(key: str, special_configs: dict, settings: dict):
    """Process or load the pipeline space configuration.

    This function checks if the given key exists in the `special_configs` dictionary.
    If it exists, it processes the associated value, which can be either a dictionary
    or a string. Based on the keys of the dictionary it decides if the pipeline_space
    have to be loaded or needs to be converted into a neps search_space structure.
    The processed pipeline space is then stored in the `settings`
    dictionary under the given key.

    Args:
        key (str): The key to check in the `special_configs` dictionary.
        special_configs (dict): The dictionary containing special configuration values.
        settings (dict): The dictionary where the processed pipeline space will be stored.

    Raises:
        TypeError: If the value associated with the key is neither a string nor a
        dictionary.
    """
    if special_configs.get(key) is not None:
        pipeline_space = special_configs[key]
        if isinstance(pipeline_space, dict):
            # determine if dict contains path_loading or the actual search space
            expected_keys = {"path", "name"}
            actual_keys = set(pipeline_space.keys())
            if expected_keys != actual_keys:
                # pipeline_space directly defined in run_args yaml
                processed_pipeline_space = pipeline_space_from_yaml(pipeline_space)
            else:
                # pipeline_space stored in a python dict, not using a yaml
                processed_pipeline_space = load_and_return_object(
                    pipeline_space["path"], pipeline_space["name"], key
                )
        elif isinstance(pipeline_space, str):
            # load yaml from path
            processed_pipeline_space = pipeline_space_from_yaml(pipeline_space)
        else:
            raise TypeError(
                f"Value for {key} must be a string or a dictionary, "
                f"but got {type(pipeline_space).__name__}."
            )
        settings[key] = processed_pipeline_space


def process_searcher(key: str, special_configs: dict, settings: dict):
    """Processes the searcher configuration and updates the settings dictionary.

    Checks if the key exists in special_configs. If found, it processes the
    value based on its type. Updates settings with the processed searcher.

    Parameters:
    key (str): Key to look up in special_configs.
    special_configs (dict): Dictionary of special configurations.
    settings (dict): Dictionary to update with the processed searcher.

    Raises:
    TypeError: If the value for the key is neither a string, Path, nor a dictionary.
    """
    if special_configs.get(key) is not None:
        searcher = special_configs[key]
        if isinstance(searcher, dict):
            # determine if dict contains path_loading or the actual searcher config
            expected_keys = {"path", "name"}
            actual_keys = set(searcher.keys())
            if expected_keys == actual_keys:
                searcher = load_and_return_object(searcher["path"], searcher["name"], key)
        elif isinstance(searcher, (str, Path)):
            pass
        else:
            raise TypeError(
                f"Value for {key} must be a string or a dictionary, "
                f"but got {type(searcher).__name__}."
            )
        settings[key] = searcher


def process_run_pipeline(key: str, special_configs: dict, settings: dict):
    """Processes the run pipeline configuration and updates the settings dictionary.

    Parameters:
    key (str): Key to look up in special_configs.
    special_configs (dict): Dictionary of special configurations.
    settings (dict): Dictionary to update with the processed function.

    Raises:
    KeyError: If required keys ('path' and 'name') are missing in the config.
    """
    if special_configs.get(key) is not None:
        config = special_configs[key]
        try:
            func = load_and_return_object(config["path"], config["name"], key)
            settings[key] = func
        except KeyError as e:
            raise KeyError(
                f"Missing key for argument {key}: {e}. Expect 'path' "
                f"and 'name' as keys when loading '{key}' "
                f"from 'run_args'"
            ) from e


def load_and_return_object(module_path: str, object_name: str, key: str) -> object:
    """Dynamically loads an object from a given module file path.

    This function attempts to dynamically import an object by its name from a specified
    module path. If the initial import fails, it retries with a '.py' extension appended
    to the path.

    Args:
        module_path (str): File system path to the Python module.
        object_name (str): Name of the object to import from the module.
        key (str): Identifier for the argument causing the error, for enhanced error
        feedback.

    Returns:
        object: The imported object from the module.

    Raises:
        ImportError: If the module or object cannot be found, with a message detailing
        the issue.
    """

    def import_object(path):
        try:
            # Convert file system path to module path, removing '.py' if present.
            module_name = (
                path[:-3].replace("/", ".")
                if path.endswith(".py")
                else path.replace("/", ".")
            )

            # Dynamically import the module.
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return None  # Failed to load module spec.
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Retrieve the object.
            imported_object = getattr(module, object_name, None)
            if imported_object is None:
                return None  # Object not found in module.
            return imported_object
        except FileNotFoundError:
            return None  # File not found.

    # Attempt to import the object using the provided path.
    imported_object = import_object(module_path)
    if imported_object is None:
        # If the object could not be imported, attempt again by appending '.py',
        # if not already present.
        if not module_path.endswith(".py"):
            module_path += ".py"
            imported_object = import_object(module_path)

        if imported_object is None:
            raise ImportError(
                f"Failed to import '{object_name}' for argument '{key}'. "
                f"Module path '{module_path}' not found or object does not "
                f"exist."
            )

    return imported_object


def load_hooks_from_config(pre_load_hooks_dict: dict) -> list:
    """Loads hook functions from a dictionary of configurations.

    Args:
        pre_load_hooks_dict (Dict): Dictionary with hook names as keys and paths as values

    Returns:
        List: List of loaded hook functions.
    """
    loaded_hooks = []
    for name, path in pre_load_hooks_dict.items():
        hook_func = load_and_return_object(path, name, PRE_LOAD_HOOKS)
        loaded_hooks.append(hook_func)
    return loaded_hooks


def check_run_args(settings: dict) -> None:
    """Validates the types of NePS configuration settings.

    Checks that each setting's value type matches its expected type. Raises
    TypeError for type mismatches.

    Args:
        settings (dict): NEPS configuration settings.

    Raises:
        TypeError: For mismatched setting value types.
    """
    # Mapping parameter names to their allowed types
    # [task_id, development_stage_id, pre_load_hooks] require special handling of type,
    # that's why they are not listed
    expected_types = {
        RUN_PIPELINE: Callable,
        ROOT_DIRECTORY: str,
        # TODO: Support CS.ConfigurationSpace for pipeline_space
        PIPELINE_SPACE: (str, dict),
        OVERWRITE_WORKING_DIRECTORY: bool,
        POST_RUN_SUMMARY: bool,
        MAX_EVALUATIONS_TOTAL: int,
        MAX_COST_TOTAL: (int, float),
        MAX_EVALUATIONS_PER_RUN: int,
        CONTINUE_UNTIL_MAX_EVALUATION_COMPLETED: bool,
        LOSS_VALUE_ON_ERROR: float,
        COST_VALUE_ON_ERROR: float,
        IGNORE_ERROR: bool,
        SEARCHER_KWARGS: dict,
    }
    for param, value in settings.items():
        if param in (DEVELOPMENT_STAGE_ID, TASK_ID):
            # this argument can be Any
            continue
        elif param == PRE_LOAD_HOOKS:
            # check if all items in pre_load_hooks are callable objects
            if not all(callable(item) for item in value):
                raise TypeError("All items in 'pre_load_hooks' must be callable.")
        elif param == SEARCHER:
            if not (isinstance(param, str) or issubclass(param, BaseOptimizer)):
                raise TypeError(
                    "Parameter 'searcher' must be a string or a class that is a subclass "
                    "of BaseOptimizer."
                )
        else:
            try:
                expected_type = expected_types[param]
            except KeyError as e:
                raise KeyError(f"{param} is not a valid argument of neps") from e
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Parameter '{param}' expects a value of type {expected_type}, got "
                    f"{type(value)} instead."
                )


def check_essential_arguments(
    run_pipeline: Callable | None,
    root_directory: str | None,
    pipeline_space: dict | None,
    max_cost_total: int | None,
    max_evaluation_total: int | None,
    searcher: BaseOptimizer | None,
    run_args: str | None,
) -> None:
    """Validates essential NEPS configuration arguments.

    Ensures 'run_pipeline', 'root_directory', 'pipeline_space', and either
    'max_cost_total' or 'max_evaluation_total' are provided for NePS execution.
    Raises ValueError with missing argument details. Additionally, checks 'searcher'
    is a BaseOptimizer if 'pipeline_space' is absent.

    Parameters:
        run_pipeline: Function for the pipeline execution.
        root_directory (str): Directory path for data storage.
        pipeline_space: search space for this run.
        max_cost_total: Max allowed total cost for experiments.
        max_evaluation_total: Max allowed evaluations.
        searcher: Optimizer for the configuration space.
        run_args: A YAML file containing the configuration settings.

    Raises:
        ValueError: Missing or invalid essential arguments.
    """
    if not run_pipeline:
        raise ValueError("'run_pipeline' is required but was not provided.")
    if not root_directory:
        raise ValueError("'root_directory' is required but was not provided.")
    if not pipeline_space and (run_args or not isinstance(searcher, BaseOptimizer)):
        # handling special case for searcher instance, in which user doesn't have to
        # provide the search_space because it's the argument of the searcher.
        raise ValueError("'pipeline_space' is required but was not provided.")

    if not max_evaluation_total and not max_cost_total:
        raise ValueError(
            "'max_evaluation_total' or 'max_cost_total' is required but "
            "both were not provided."
        )


def check_double_reference(
    func: Callable, func_arguments: dict, yaml_arguments: dict
) -> Any:
    """Checks if no argument is defined both via function arguments and YAML.

    Parameters:
    - func (Callable): The function to check arguments against.
    - func_arguments (Dict): A dictionary containing the provided arguments to the
    function and their values.
    - yaml_arguments (Dict): A dictionary containing the arguments provided via a YAML
    file.

    Raises:
    - ValueError: If any provided argument is defined both via function arguments and the
    YAML file.
    """
    sig = inspect.signature(func)

    for name, param in sig.parameters.items():
        if param.default != func_arguments[name]:
            if name == RUN_ARGS:
                # Ignoring run_args argument
                continue
            if name in yaml_arguments:
                raise ValueError(
                    f"Conflict for argument '{name}': Argument is defined both via "
                    f"function arguments and YAML, which is not allowed."
                )
