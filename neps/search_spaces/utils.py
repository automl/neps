from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import yaml

from neps.search_spaces.hyperparameters import (
    CategoricalParameter,
    ConstantParameter,
    FloatParameter,
    IntegerParameter,
)

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from neps.search_spaces.hyperparameters import Parameter

logger = logging.getLogger("neps")


@overload
def convert_scientific_notation(
    value: str | int | float, *, show_usage_flag: Literal[False] = False
) -> float: ...


@overload
def convert_scientific_notation(
    value: str | int | float, *, show_usage_flag: Literal[True]
) -> tuple[float, bool]: ...


def convert_scientific_notation(
    value: str | int | float, *, show_usage_flag: bool = False
) -> float | tuple[float, bool]:
    """Convert a given value to float if it's a string that matches scientific e notation.

    This is especially useful for numbers like "3.3e-5" which YAML parsers may not
    directly interpret as floats.

    If the 'show_usage_flag' is set to True, the function returns a tuple of the float
    conversion and a boolean flag indicating whether scientific notation was detected.

    Args:
        value (str | int | float): The value to convert. Can be an integer, float,
                                   or a string representing a number, possibly in
                                   scientific notation.
        show_usage_flag (bool): Optional; defaults to False. If True, the function
                                also returns a flag indicating whether scientific
                                notation was detected in the string.

    Returns:
        float: The value converted to float if 'show_usage_flag' is False.
        (float, bool): A tuple containing the value converted to float and a flag
                       indicating scientific notation detection if 'show_usage_flag'
                       is True.

    Raises:
        ValueError: If the value is a string and does not represent a valid number.
    """
    e_notation_pattern = r"^-?\d+(\.\d+)?[eE]-?\d+$"

    flag = False  # Flag if e notation was detected

    if isinstance(value, str):
        # Remove all whitespace from the string
        value_no_space = value.replace(" ", "")

        # check for e notation
        if re.match(e_notation_pattern, value_no_space):
            flag = True

    if show_usage_flag is True:
        return float(value), flag

    return float(value)


class SearchSpaceFromYamlFileError(Exception):
    """Exception raised for errors occurring during the initialization of the search space
    from a YAML file.

    Attributes:
        exception_type: The type of the original exception.
        message: A detailed message that includes the type of the original exception and
            the error description.
    """

    def __init__(self, exception: Exception) -> None:
        """Initializes the SearchSpaceFromYamlFileError with the original exception.

        Args:
            exception: The original exception that was raised during the initialization of
                the search space from the YAML file.
        """
        self.exception_type: str = type(exception).__name__
        self.message: str = (
            f"Error occurred during initialization of search space from "
            f"YAML file.\n {self.exception_type}: {exception}"
        )
        super().__init__(self.message)


def deduce_type(
    name: str, details: dict[str, str | int | float] | str | int | float
) -> str:
    """Deduces the parameter type from details.

    Args:
        name: The name of the parameter.
        details: A dictionary containing parameter specifications or
            a direct value (string, integer, or float).

    Returns:
        The deduced parameter type ('int', 'float', 'categorical', or 'constant').

    Raises:
        TypeError: If the type cannot be deduced or the details don't align with expected
                constraints.
    """
    if isinstance(details, (str, int, float)):
        return "const"

    if isinstance(details, dict):
        if "type" in details:
            param_type = details.pop("type")
            assert isinstance(param_type, str)
            return param_type.lower()

        return deduce_param_type(name, details)

    raise TypeError(
        f"Unable to deduce parameter type for '{name}' with details '{details}'."
    )


def deduce_param_type(name: str, details: dict[str, int | str | float]) -> str:
    """Deduces the parameter type based on the provided details.

    The function interprets the 'details' dictionary to determine the parameter type.
    The dictionary should include key-value pairs that describe the parameter's
    characteristics, such as lower, upper and choices.


    Args:
        name: The name of the parameter.
        details: A dictionary containing parameter specifications.

    Returns:
        str: The deduced parameter type ('int', 'float' or 'categorical').

    Raises:
        TypeError: If the parameter type cannot be deduced from the details, or if the
        provided details have inconsistent types for expected keys.
    """
    # check for int and float conditions
    if "lower" in details and "upper" in details:
        # Determine if it's an integer or float range parameter
        if isinstance(details["lower"], int) and isinstance(details["upper"], int):
            param_type = "int"
        elif isinstance(details["lower"], float) and isinstance(details["upper"], float):
            param_type = "float"
        else:
            try:
                details["lower"], flag_lower = convert_scientific_notation(
                    details["lower"], show_usage_flag=True
                )
                details["upper"], flag_upper = convert_scientific_notation(
                    details["upper"], show_usage_flag=True
                )
            except ValueError as e:
                raise TypeError(
                    f"Inconsistent types for 'lower' and 'upper' in '{name}'. "
                    f"Both must be either integers or floats."
                ) from e

            # check if one value is e notation and if so convert it to float
            if flag_lower or flag_upper:
                logger.info(
                    f"Because of e notation, Parameter {name} gets "
                    f"interpreted as float"
                )
                param_type = "float"
            else:
                raise TypeError(
                    f"Inconsistent types for 'lower' and 'upper' in '{name}'. "
                    f"Both must be either integers or floats."
                )
    # check for categorical condition
    elif "choices" in details:
        param_type = "categorical"
    else:
        raise KeyError(
            f"Unable to deduce parameter type from {name} "
            f"with details {details}\n"
            "Supported parameters:\n"
            "Float and Integer: Expected keys: 'lower', 'upper'\n"
            "Categorical: Expected keys: 'choices'\n"
        )
    return param_type


def formatting_int(name: str, details: dict[str, str | int | float]) -> dict:
    """Converts scientific notation values to integers.

    This function converts the 'lower' and 'upper' bounds, as well as the 'default'
    value (if present), from scientific notation to integers.

    Args:
        name: The name of the integer parameter.
        details: A dictionary containing the parameter's specifications.
            Expected keys include 'lower', 'upper', and optionally 'default'.

    Raises:
        TypeError: If 'lower', 'upper', or 'default' cannot be converted from scientific
                   notation to integers.

    Returns:
        The dictionary with the converted integer parameter details.
    """
    if not isinstance(details["lower"], int) or not isinstance(details["upper"], int):
        try:
            # for numbers like 1e2 and 10^
            lower, flag_lower = convert_scientific_notation(
                details["lower"], show_usage_flag=True
            )
            upper, flag_upper = convert_scientific_notation(
                details["upper"], show_usage_flag=True
            )
            # check if one value format is e notation and if it's an integer
            if flag_lower or flag_upper:
                if lower == int(lower) and upper == int(upper):
                    details["lower"] = int(lower)
                    details["upper"] = int(upper)
                else:
                    raise TypeError()
            else:
                raise TypeError()
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"'lower' and 'upper' must be integer for " f"integer parameter '{name}'."
            ) from e
    if "default" in details and not isinstance(details["default"], int):
        try:
            # convert value can raise ValueError
            default = convert_scientific_notation(details["default"])
            if default == int(default):
                details["default"] = int(default)
            else:
                raise TypeError()  # type of value is not int
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"default value {details['default']} "
                f"must be integer for integer parameter {name}"
            ) from e
    return details


def formatting_float(name: str, details: dict[str, str | int | float]) -> dict:
    """Converts scientific notation values to floats.

    This function converts the 'lower' and 'upper' bounds, as well as the 'default'
    value (if present), from scientific notation to floats.

    Args:
        name: The name of the float parameter.
        details: A dictionary containing the parameter's specifications. Expected keys
                 include 'lower', 'upper', and optionally 'default'.

    Raises:
        TypeError: If 'lower', 'upper', or 'default' cannot be converted from scientific
                   notation to floats.

    Returns:
        The dictionary with the converted float parameter details.
    """
    if not isinstance(details["lower"], float) or not isinstance(details["upper"], float):
        try:
            # for numbers like 1e-5 and 10^
            details["lower"] = convert_scientific_notation(details["lower"])
            details["upper"] = convert_scientific_notation(details["upper"])
        except ValueError as e:
            raise TypeError(
                f"'lower' and 'upper' must be float for " f"float parameter '{name}'."
            ) from e
    if "default" in details and not isinstance(details["default"], float):
        try:
            details["default"] = convert_scientific_notation(details["default"])
        except ValueError as e:
            raise TypeError(
                f" default'{details['default']}' must be float for float "
                f"parameter {name} "
            ) from e
    return details


def formatting_cat(name: str, details: dict[str, list | str | int | float]) -> dict:
    """This function ensures that the 'choices' key in the details is a list and attempts
    to convert any elements expressed in scientific notation to floats. It also handles
    the 'default' value, converting it from scientific notation if necessary.

    Args:
        name: The name of the categorical parameter.
        details: A dictionary containing the parameter's specifications. The required key
                 is 'choices', which must be a list. The 'default' key is optional.

    Raises:
        TypeError: If 'choices' is not a list.

    Returns:
        The validated and possibly converted categorical parameter details.
    """
    if not isinstance(details["choices"], list):
        raise TypeError(f"The 'choices' for '{name}' must be a list.")

    for i, element in enumerate(details["choices"]):
        try:
            converted_value, e_flag = convert_scientific_notation(
                element, show_usage_flag=True
            )

            if e_flag:
                # Replace the element at the same position
                details["choices"][i] = converted_value
        except ValueError:
            pass  # If a ValueError occurs, simply continue to the next element

    if "default" in details:
        e_flag = False
        extracted_default = details["default"]
        if not isinstance(extracted_default, (str, int, float)):
            raise TypeError(
                f"The 'default' value for '{name}' must be a string, integer, or float."
                f" Got {type(extracted_default).__name__}."
            )

        # check if e notation, if then convert to number
        with contextlib.suppress(ValueError):
            default, e_flag = convert_scientific_notation(
                extracted_default, show_usage_flag=True
            )

        if e_flag is True:
            details["default"] = default

    return details


def formatting_const(details: str | int | float) -> str | int | float:
    """Validates and converts a constant parameter.

    This function checks if the 'details' parameter contains a value expressed in
    scientific notation and converts it to a float. It ensures that the input
    is appropriately formatted, either as a string, integer, or float.

    Args:
        details: A constant parameter that can be a string, integer, or float.
                 If the value is in scientific notation, it will be converted to a float.

    Returns:
        The validated and possibly converted constant parameter.
    """
    e_flag = False
    with contextlib.suppress(ValueError):
        # if the value is not able to convert to float a ValueError get raised by
        # convert_scientific_notation function
        converted_value, e_flag = convert_scientific_notation(
            details, show_usage_flag=True
        )

    if e_flag:
        details = converted_value

    return details


def pipeline_space_from_yaml(  # noqa: C901
    config: str | Path | dict,
) -> dict[str, Parameter]:
    """Reads configuration details from a YAML file or a dictionary and constructs a
    pipeline space dictionary.

    Args:
        config: Path to the YAML file or a dictionary containing parameter configurations.

    Returns:
        A dictionary where keys are parameter names and values are parameter objects.

    Raises:
        SearchSpaceFromYamlFileError: Raised if there are issues with the YAML file's
            format, contents, or if the dictionary is invalid.
    """
    try:
        if isinstance(config, (str, Path)):
            # try to load the YAML file
            try:
                yaml_file_path = Path(config)
                with yaml_file_path.open("r") as file:
                    config = yaml.safe_load(file)
                if not isinstance(config, dict):
                    raise ValueError(
                        "The loaded pipeline_space is not a valid dictionary. Please "
                        "ensure that you use a proper structure. See the documentation "
                        "for more details."
                    )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Unable to find the specified file for 'pipeline_space' at "
                    f"'{config}'. Please verify the path specified in the "
                    f"'pipeline_space' argument and try again."
                ) from e
            except yaml.YAMLError as e:
                raise ValueError(f"The file at {config} is not a valid YAML file.") from e

        pipeline_space: dict[str, Parameter] = {}

        for name, details in config.items():
            param_type = deduce_type(name, details)

            if param_type in ("int", "integer"):
                formatted_details = formatting_int(name, details)
                pipeline_space[name] = IntegerParameter(**formatted_details)
            elif param_type == "float":
                formatted_details = formatting_float(name, details)
                pipeline_space[name] = FloatParameter(**formatted_details)
            elif param_type in ("cat", "categorical"):
                formatted_details = formatting_cat(name, details)
                pipeline_space[name] = CategoricalParameter(**formatted_details)
            elif param_type == "const":
                const_details = formatting_const(details)
                pipeline_space[name] = ConstantParameter(const_details)
            else:
                # Handle unknown parameter type
                raise TypeError(
                    f"Unsupported parameter with details: {details} for '{name}'.\n"
                    f"Supported Types for argument type are:\n"
                    "For integer parameter: int, integer\n"
                    "For float parameter: float\n"
                    "For categorical parameter: cat, categorical\n"
                    "Constant parameter was not detect\n"
                )
    except (KeyError, TypeError, ValueError, FileNotFoundError) as e:
        raise SearchSpaceFromYamlFileError(e) from e

    return pipeline_space


def pipeline_space_from_configspace(
    configspace: ConfigurationSpace,
) -> dict[str, Parameter]:
    """Constructs the [`Parameter`][neps.search_spaces.parameter.Parameter] objects
    from a [`ConfigurationSpace`][ConfigSpace.configuration_space.ConfigurationSpace].

    Args:
        configspace: The configuration space to construct the pipeline space from.

    Returns:
        A dictionary where keys are parameter names and values are parameter objects.
    """
    import ConfigSpace as CS

    pipeline_space = {}
    parameter: Parameter
    if any(configspace.get_conditions()) or any(configspace.get_forbiddens()):
        raise NotImplementedError(
            "The ConfigurationSpace has conditions or forbidden clauses, "
            "which are not supported by neps."
        )

    for hyperparameter in configspace.get_hyperparameters():
        if isinstance(hyperparameter, CS.Constant):
            parameter = ConstantParameter(value=hyperparameter.value)
        elif isinstance(hyperparameter, CS.CategoricalHyperparameter):
            parameter = CategoricalParameter(
                hyperparameter.choices,
                default=hyperparameter.default_value,
            )
        elif isinstance(hyperparameter, CS.OrdinalHyperparameter):
            parameter = CategoricalParameter(
                hyperparameter.sequence,
                default=hyperparameter.default_value,
            )
        elif isinstance(hyperparameter, CS.UniformIntegerHyperparameter):
            parameter = IntegerParameter(
                lower=hyperparameter.lower,
                upper=hyperparameter.upper,
                log=hyperparameter.log,
                default=hyperparameter.default_value,
            )
        elif isinstance(hyperparameter, CS.UniformFloatHyperparameter):
            parameter = FloatParameter(
                lower=hyperparameter.lower,
                upper=hyperparameter.upper,
                log=hyperparameter.log,
                default=hyperparameter.default_value,
            )
        else:
            raise ValueError(f"Unknown hyperparameter type {hyperparameter}")
        pipeline_space[hyperparameter.name] = parameter
    return pipeline_space
