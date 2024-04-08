import importlib.util
import logging
import sys
import yaml
from collections.abc import Callable
from neps.optimizers.base_optimizer import BaseOptimizer

logger = logging.getLogger("neps")

# Define the name of the arguments as variables for easier code maintenance
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


def get_run_args_from_yaml(path):
    """
    Load and validate NEPS run arguments from a specified YAML configuration file
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
        SEARCHER_PATH,
    ]

    # Flatten the YAML file's structure to separate flat parameters (flat_config) and
    # those needing special handling (special_configs).
    flat_config, special_configs = extract_leaf_keys(config)

    # Check if flatten dict (flat_config) just contains the expected parameters
    for parameter, value in flat_config.items():
        if parameter in expected_parameters:
            settings[parameter] = value
        else:
            raise KeyError(f"Parameter '{parameter}' is not an argument of neps.run().")

    # Process complex configurations (e.g., 'pipeline_space', 'searcher') and integrate
    # them into 'settings'.
    handle_special_argument_cases(settings, special_configs)

    # check if all provided arguments have legal types
    check_run_args(settings)

    logger.debug(
        f"'run_args' are extracted and type-tested from the referenced YAML file. "
        f"These arguments will now be overwritten: {settings}."
    )

    return settings


def config_loader(path):
    """
    Loads and parses a YAML configuration file. Checks if the loaded YAML starts with
    'run_args' key.

    Args:
        path (str): The filesystem path to the YAML file to be loaded.

    Returns:
        dict: The parsed YAML file as a dictionary, specifically the content under
        'run_args' key.

    Raises:
        ValueError: If the file cannot be parsed as YAML.
        KeyError: If the 'run_args' key is not found at the top level of the YAML file.
    """
    try:
        with open(path) as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"The file at {path} is not a valid YAML file.") from e

    # Check if 'run_args' is the top-level key in the loaded YAML
    if 'run_args' not in config:
        raise KeyError(f"The 'run_args' key is missing at the top level of the YAML "
                       f"file: {path}")

    return config['run_args']


def extract_leaf_keys(d, special_keys=None):
    """
    Recursive function to extract leaf keys and their values from a nested dictionary.
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
                              that require special processing.

    Returns:
    - None: The function modifies 'settings' in place.

    """
    # Load the value of each key from a dictionary specifying "path" and "name".
    process_config_key(
        settings, special_configs, [PIPELINE_SPACE, SEARCHER, RUN_PIPELINE]
    )

    if special_configs[SEARCHER_KWARGS] is not None:
        configs = {}
        # Check if values of keys is not None and then add the dict to settings
        # Increase flexibility to let value of a key in yaml file empty
        for key, value in special_configs[SEARCHER_KWARGS].items():
            if value is not None:
                configs[key] = value
        if len(configs) != 0:
            settings[SEARCHER_KWARGS] = configs

    if special_configs[PRE_LOAD_HOOKS] is not None:
        # Loads the pre_load_hooks functions and add them in a list to settings.
        settings[PRE_LOAD_HOOKS] = load_hooks_from_config(special_configs[PRE_LOAD_HOOKS])


def process_config_key(settings, special_configs, keys):
    """
    Enhance 'settings' by adding keys and their corresponding values or loaded objects
    from 'special_configs'. Keys in 'special_configs' are processed to directly insert
    their values into 'settings' or to load functions/objects using 'path' and 'name'.
    Key handling varies: 'RUN_PIPELINE' requires a dictionary defining a loadable function
    , whereas other keys may accept either strings or dictionaries

    Parameters:
    - settings (dict): Dictionary to update.
    - special_configs (dict): Contains keys and values for processing.
    - keys (list): List of keys to process in 'special_configs'.

    Raises:
    - KeyError: Missing 'path'/'name' for dictionaries.
    - TypeError: Incorrect type for key's value; RUN_PIPELINE must be a dict,
    others can be dict or string.
    """
    for key in keys:
        if special_configs.get(key) is not None:
            value = special_configs[key]
            if isinstance(value, str) and key != RUN_PIPELINE:
                # pipeline_space and searcher can also be a string
                settings[key] = value
            elif isinstance(value, dict):
                # dict that should contain 'path' and 'name' for loading value
                # (function, dict, class)
                try:
                    func = load_and_return_object(value["path"], value["name"])
                    settings[key] = func
                except KeyError as e:
                    raise KeyError(
                        f"Missing key for argument {key}: {e}. Expect 'path' "
                        f"and 'name' as keys when loading '{key}' "
                        f"from 'run_args'"
                    ) from e
            else:
                if key == RUN_PIPELINE:
                    raise TypeError(
                        f"Value for {key} must be a dictionary, but got "
                        f"{type(value).__name__}."
                    )
                else:
                    raise TypeError(
                        f"Value for {key} must be a string or a dictionary, "
                        f"but got {type(value).__name__}."
                    )


def load_and_return_object(module_path, object_name):
    """
    Dynamically imports an object from a specified module file path.

    This function can import various types of objects from a module, including
    dictionaries, class instances, or functions. It does so by specifying the module's
    file system path and the object's name within that module.

    Args:
        module_path (str): The file system path to the Python module.
        object_name (str): The name of the object to import from the module.

    Returns:
        object: The imported object, which can be of any type (e.g., dict, function,
        class).

    Raises:
        ImportError: If the module or object cannot be found.
    """
    try:
        # Convert file system path to module path.
        module_name = module_path.replace("/", ".").rstrip(".py")

        # Dynamically import the module.
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Retrieve the object.
        imported_object = getattr(module, object_name)

    except FileNotFoundError as exc:
        raise ImportError(f"Module path '{module_path}' not found.") from exc
    except AttributeError as exc:
        raise ImportError(
            f"Object '{object_name}' not found in module '{module_name}'."
        ) from exc

    return imported_object


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
            hook_func = load_and_return_object(hook_config["path"], hook_config["name"])
            loaded_hooks.append(hook_func)
        else:
            raise KeyError(
                f"Expected keys 'path' and 'name' for hook, but got: " f"{hook_config}."
            )
    return loaded_hooks


def check_run_args(settings):
    """
    Validates the types of NePS configuration settings.

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
        SEARCHER_PATH: str,
        SEARCHER_KWARGS: dict,
    }
    for param, value in settings.items():
        if param == DEVELOPMENT_STAGE_ID or param == TASK_ID:
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
    run_pipeline,
    root_directory,
    pipeline_space,
    max_cost_total,
    max_evaluation_total,
    searcher,
    run_args,
):
    """
    Validates essential NEPS configuration arguments.

    Ensures 'run_pipeline', 'root_directory', 'pipeline_space', and either
    'max_cost_total' or 'max_evaluation_total' are provided for NEPS execution.
    Raises ValueError with missing argument details. Additionally, checks 'searcher'
    is a BaseOptimizer if 'pipeline_space' is absent.

    Parameters:
        run_pipeline: Function for the pipeline execution.
        root_directory (str): Directory path for data storage.
        pipeline_space: search space for this run.
        max_cost_total: Max allowed total cost for experiments.
        max_evaluation_total: Max allowed evaluations.
        searcher: Optimizer for the configuration space.

    Raises:
        ValueError: Missing or invalid essential arguments.
    """
    if not run_pipeline:
        raise ValueError("'run_pipeline' is required but was not provided.")
    if not root_directory:
        raise ValueError("'root_directory' is required but was not provided.")
    if not pipeline_space:
        # handling special case for searcher instance, in which user doesn't have to
        # provide the search_space because it's the argument of the searcher.
        if run_args or not isinstance(searcher, BaseOptimizer):
            raise ValueError("'pipeline_space' is required but was not provided.")

    if not max_evaluation_total and not max_cost_total:
        raise ValueError(
            "'max_evaluation_total' or 'max_cost_total' is required but "
            "both were not provided."
        )