import importlib.util
import logging
import sys
from collections.abc import Callable
from neps.optimizers.base_optimizer import BaseOptimizer
import yaml

logger = logging.getLogger("neps")

# Define the name of the arguments as variable for easier code maintenance
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
SEARCHER_PATH = "searcher_path"
PRE_LOAD_HOOKS = "pre_load_hooks"
SEARCHER_KWARGS = "searcher_kwargs"
MAX_EVALUATIONS_PER_RUN = "max_evaluations_per_run"

EXPECTED_PARAMETERS = [
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
    SEARCHER_PATH,
]


def get_run_args_from_yaml(path):
    """
    Loads NEPS optimization settings from a YAML file and validates them.

    Ensures all settings are valid NEPS.run() arguments with correct structure. Extracts
    special configurations like 'searcher_kwargs', 'run_pipeline', and 'pre_load_hooks',
    converting them as necessary. Raises an error for unrecognized parameters or incorrect
    file structure.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: Validated and processed run arguments.

    Raises:
        KeyError: For invalid parameters or missing 'path'/'name' in special
        configurations.
    """
    # Load the YAML configuration file
    config = config_loader(path)

    # Initialize an empty dictionary to hold the extracted settings
    settings = {}

    # Define the allowed parameters based on the arguments of neps.run(), that have a
    # simple value like a string or int type
    # [searcher_kwargs, run_pipeline, preload_hooks, pipeline_space, searcher] require
    # special handling due to their
    # values necessitating distinct treatment, that's why they are not listed
    EXPECTED_PARAMETERS = [
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
        SEARCHER_PATH,
    ]

    # Flatten yaml file and ignore hierarchical structure, only consider parameters(keys)
    # with an explicit value
    flat_config, special_configs = extract_leaf_keys(config)

    # Check if flatten dict just contains neps arguments
    for parameter, value in flat_config.items():
        if parameter in EXPECTED_PARAMETERS:
            settings[parameter] = value
        else:
            raise KeyError(f"Parameter '{parameter}' is not an argument of neps.run().")

    handle_special_argument_cases(settings, special_configs)

    # check if arguments have legal types of provided
    check_run_args(settings)

    logger.debug(
        f"'run_args' are extracted and type-tested from the referenced YAML file. "
        f"These arguments will now be overwritten: {settings}."
    )

    return settings


def config_loader(path):
    """
    Loads and parses a YAML configuration file.

    Args:
        path (str): The filesystem path to the YAML file to be loaded.

    Returns:
        dict: The parsed YAML file as a dictionary.

    Raises:
        ValueError: If the file cannot be parsed as YAML.
    """
    try:
        with open(path) as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"The file at {str(path)} is not a valid YAML file.") from e
    return config


def extract_leaf_keys(d, special_keys=None):
    """
    Recursive function to extract leaf keys and their values from a nested dictionary.
    Special keys ('searcher_kwargs', 'run_pipeline', 'pre_load_hooks') are also
    extracted if present and their corresponding values (dict) at any level in the
    nested
    structure.

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
            PIPELINE_SPACE: None
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


def handle_special_argument_cases(settings, special_configs):
    """
    Process and integrate special configuration cases into the 'settings' dictionary.

    This function updates 'settings' with values from 'special_configs'. It handles
    specific keys that require more complex processing, such as 'pipeline_space' and
    'searcher', which may need to load a function/dict from paths. It also manages nested
    configurations like 'searcher_kwargs' and 'pre_load_hooks' which need individual
    processing or function loading.

    Parameters:
    - settings (dict): The dictionary to be updated with processed configurations.
    - special_configs (dict): A dictionary containing configuration keys and values
                              that require special handling.

    Returns:
    - None: The function modifies 'settings' in place.

    Raises:
    - KeyError: If essential keys are missing for loading functions.
    """
    # handle arguments that differ from a simple/single value/str
    process_config_key(settings, special_configs, PIPELINE_SPACE)
    process_config_key(settings, special_configs, SEARCHER)

    # handle arguments that differ from a simple/single value/str
    if special_configs[SEARCHER_KWARGS] is not None:
        configs = {}

        # important for handling yaml configs that provide the key but not a value in it,
        # increase usability of a yaml template where just values but not the keys have
        # to be removed
        for key, value in special_configs[SEARCHER_KWARGS].items():
            if value is not None:
                configs[key] = value
        if len(configs) != 0:
            settings[SEARCHER_KWARGS] = configs

    if special_configs[RUN_PIPELINE] is not None:
        run_pipeline_dict = special_configs[RUN_PIPELINE]
        # load function via path and name
        try:
            func = load_and_return_function(
                run_pipeline_dict["path"], run_pipeline_dict["name"]
            )
            settings[RUN_PIPELINE] = func
        except KeyError as e:
            raise KeyError(
                f"Missing key for argument run_pipeline: {e}. Expect 'path "
                f"and 'name' as keys when loading 'run_pipeline' "
                f"from 'run_args'"
            ) from e

    if special_configs[PRE_LOAD_HOOKS] is not None:
        # Loads functions and returns them in a list.
        settings[PRE_LOAD_HOOKS] = load_hooks_from_config(
            special_configs[PRE_LOAD_HOOKS]
        )


def process_config_key(settings, special_configs, key):
    """Process a specific key from 'special_configs' and update 'settings' accordingly.

    This function expects 'special_configs[key]' to either be a string or a dictionary.
    If it's a string, it's simply added to 'settings'. If it's a dictionary, the function
    will attempt to load another function or object specified by the 'path' and 'name'
    keys within the dictionary.

    Parameters:
    - settings (dict): The dictionary to be updated with processed configurations.
    - special_configs (dict): A dictionary containing configuration keys and values.
    - key (str): The key in 'special_configs' that needs to be processed.

    Raises:
    - KeyError: If the expected keys ('path' and 'name') are missing in the value
    dictionary.
    - TypeError: If the value associated with 'key' is neither a string nor a dictionary.
"""
    if special_configs.get(key) is not None:
        value = special_configs[key]
        if isinstance(value, str):
            settings[key] = value
        elif isinstance(value, dict):
            try:
                func = load_and_return_function(value["path"], value["name"])
                settings[key] = func
            except KeyError as e:
                raise KeyError(
                    f"Missing key for argument {key}: {e}. Expect 'path' "
                    f"and 'name' as keys when loading '{key}' "
                    f"from 'run_args'"
                ) from e
        else:
            raise TypeError(f"Value for {key} must be a string or a dictionary, but "
                            f"got {type(value).__name__}.")


def load_and_return_function(module_path, function_name):
    """
    Dynamically imports a function from a specified module file path.

    Args:
        module_path (str): The file system path to the Python module.
        function_name (str): The name of the function to import from the module.

    Returns:
        function: The imported function object.

    Raises:
        ImportError: If the module or function cannot be found.
    """
    try:
        # Convert file path to module path if necessary
        module_name = module_path.replace("/", ".").rstrip(".py")

        # Dynamically import the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Retrieve the function
        func = getattr(module, function_name)

    except FileNotFoundError as exc:
        raise ImportError(f"Module path '{module_path}' not found.") from exc

    except AttributeError as exc:
        raise ImportError(
            f"Function '{function_name}' not found in module {module_name}." f""
        ) from exc

    return func


def load_hooks_from_config(pre_load_hooks_dict):
    """
    Loads hook functions from configurations.

    Iterates through a dictionary of pre-load hooks, dynamically imports each
    specified function by its 'path' and 'name', and accumulates the loaded
    functions into a list.

    Args:
        pre_load_hooks_dict (dict): Dictionary of hook configurations, each
        containing 'path' and 'name' keys for the hook function.

    Returns:
        list: List of loaded hook functions.

    Raises:
        KeyError: If any hook configuration lacks 'path' or 'name' keys.
    """
    loaded_hooks = []
    for hook_config in pre_load_hooks_dict.values():
        if "path" in hook_config and "name" in hook_config:
            hook_func = load_and_return_function(hook_config["path"], hook_config["name"])
            loaded_hooks.append(hook_func)
        else:
            raise KeyError(
                f"Expected keys 'path' and 'name' for hook, but got: " f"{hook_config}."
            )
    return loaded_hooks


def check_run_args(settings):
    """
    Validates the types of NEPS configuration settings.

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
    ALLOWED_TYPES = {
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
        SEARCHER: (str, Callable),
        SEARCHER_PATH: str,
        SEARCHER_KWARGS: dict,
    }
    for param, value in settings.items():
        if param == DEVELOPMENT_STAGE_ID or param == TASK_ID:
            # this argument can be Any
            continue
        if param == PRE_LOAD_HOOKS:
            # check if all items in pre_load_hooks are callable objects
            if not all(callable(item) for item in value):
                raise TypeError("All items in 'pre_load_hooks' must be callable.")
        else:
            expected_type = ALLOWED_TYPES[param]
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Parameter '{param}' expects a value of type {expected_type}, got "
                    f"{type(value)} instead."
                )


def check_essential_arguments(
    run_pipeline, root_directory, pipeline_space, max_cost_total, max_evaluation_total,
    searcher):
    """
    Verifies that essential NEPS arguments are provided.

    Checks for the presence of 'run_pipeline', 'root_directory', 'pipeline_space',
    and at least one of 'max_cost_total' or 'max_evaluation_total'. If any of these
    essential arguments are missing, raises a ValueError indicating the specific
    missing argument.

    Args:
        run_pipeline: The main function to run the pipeline.
        root_directory (str): The root directory for storing experiment data.
        pipeline_space: The space of the pipeline configurations.
        max_cost_total: The maximum total cost allowed for the experiments.
        max_evaluation_total: The maximum number of evaluations allowed.
        searcher: The optimizer to use for the optimization procedure.

    Raises:
        ValueError: If any of the essential arguments are missing or both
                    'max_cost_total' and 'max_evaluation_total' are not provided.
    """
    if not run_pipeline:
        raise ValueError("'run_pipeline' is required but was not provided.")
    if not root_directory:
        raise ValueError("'root_directory' is required but was not provided.")
    if not pipeline_space:
        if not isinstance(searcher, BaseOptimizer):
            raise ValueError("'pipeline_space' is required but was not provided.")
    if not max_evaluation_total and not max_cost_total:
        raise ValueError(
            "'max_evaluation_total' or 'max_cost_total' is required but "
            "both were not provided."
        )
