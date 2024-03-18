import importlib.util
import logging
import sys
from collections.abc import Callable

import yaml

logger = logging.getLogger("neps")


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

    # Define the allowed parameters based on the arguments of neps.run()
    allowed_parameters = [
        "pipeline_space",
        "root_directory",
        "max_evaluations_total",
        "max_cost_total",
        "overwrite_working_directory",
        "post_run_summary",
        "development_stage_id",
        "task_id",
        "max_evaluations_per_run",
        "continue_until_max_evaluation_completed",
        "loss_value_on_error",
        "cost_value_on_error",
        "ignore_errors",
        "searcher",
        "searcher_path",
    ]

    # Initialize an empty dictionary to hold the extracted settings
    settings = {}

    # Flatten yaml file and ignore hierarchical structure, only consider parameters(keys)
    # with a value
    flat_config, special_configs = extract_leaf_keys(config)

    # Check if just neps arguments are provided
    for parameter, value in flat_config.items():
        if parameter in allowed_parameters:
            settings[parameter] = value
        else:
            raise KeyError(f"Parameter '{parameter}' is not an argument of neps.run().")

    # handle arguments that differ from a simple/single value/str
    if special_configs["searcher_kwargs"] is not None:
        configs = {}

        # important for handling yaml configs that provide the key but not a value in it,
        # increase usability of a yaml template where just values but not the keys have
        # to be removed
        for key, value in special_configs["searcher_kwargs"].items():
            if value is not None:
                configs[key] = value
        if len(configs) != 0:
            settings["searcher_kwargs"] = configs

    if special_configs["run_pipeline"] is not None:
        run_pipeline_dict = special_configs["run_pipeline"]
        # load function via path and name
        try:
            func = load_and_return_function(
                run_pipeline_dict["path"], run_pipeline_dict["name"]
            )
            settings["run_pipeline"] = func
        except KeyError as e:
            raise KeyError(
                f"Missing key for argument run_pipeline: {e}. Expect 'path "
                f"and 'name' as keys when loading 'run_pipeline' "
                f"from 'run_args'"
            ) from e

    if special_configs["pre_load_hooks"] is not None:
        # Loads functions and returns them in a list.
        settings["pre_load_hooks"] = load_hooks_from_config(
            special_configs["pre_load_hooks"]
        )

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
            "searcher_kwargs": None,
            "run_pipeline": None,
            "pre_load_hooks": None,
        }

    leaf_keys = {}
    for k, v in d.items():
        if k in special_keys and isinstance(v, dict):
            special_keys[k] = v
        elif isinstance(v, dict):
            # Recursively call to explore nested dictionaries
            nested_leaf_keys, _ = extract_leaf_keys(v, special_keys)
            leaf_keys.update(nested_leaf_keys)
        elif v is not None and v != "None":
            leaf_keys[k] = v
    return leaf_keys, special_keys


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
    allowed_types = {
        "run_pipeline": Callable,
        "root_directory": str,
        "pipeline_space": str,
        "max_evaluations_total": int,
        "max_cost_total": (int, float),
        "overwrite_working_directory": bool,
        "post_run_summary": bool,
        "max_evaluations_per_run": int,
        "continue_until_max_evaluation_completed": bool,
        "loss_value_on_error": float,
        "cost_value_on_error": float,
        "ignore_errors": bool,
        "searcher": str,
        "searcher_path": str,
        "searcher_kwargs": dict,
    }
    for param, value in settings.items():
        if param == "development_stage_id" or param == "task_id":
            # this argument can be Any
            continue
        if param == "pre_load_hooks":
            # check if all items in pre_load_hooks are callable objects
            if not all(callable(item) for item in value):
                raise TypeError("All items in 'pre_load_hooks' must be callable.")
        else:
            expected_type = allowed_types[param]
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

    Raises:
        ValueError: If any of the essential arguments are missing or both
                    'max_cost_total' and 'max_evaluation_total' are not provided.
    """
    if not run_pipeline:
        raise ValueError("'run_pipeline' is required but was not provided.")
    if not root_directory:
        raise ValueError("'root_directory' is required but was not provided.")
    if not pipeline_space:
        if searcher is not Callable:
            raise ValueError("'pipeline_space' is required but was not provided.")
    if not max_evaluation_total and not max_cost_total:
        raise ValueError(
            "'max_evaluation_total' or 'max_cost_total' is required but "
            "both were not provided."
        )

