from __future__ import annotations

import logging
import re
from neps.search_spaces.hyperparameters import ConstantParameter, CategoricalParameter,\
    FloatParameter, IntegerParameter

logger = logging.getLogger(__name__)


def convert_scientific_notation(value: str | int | float, show_usage_flag=False) \
        -> float | (float, bool):
    """
    Convert a given value to a float if it's a string that matches scientific e notation.
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
    else:
        return float(value)


class SearchSpaceFromYamlFileError(Exception):
    """
    Exception raised for errors occurring during the initialization of the search space
    from a YAML file.

    Attributes:
        exception_type (str): The type of the original exception.
        message (str): A detailed message that includes the type of the original exception
                       and the error description.

    Args:
        exception (Exception): The original exception that was raised during the
                                initialization of the search space from the YAML file.

    Example Usage:
        try:
            # Code to initialize search space from YAML file
        except (KeyError, TypeError, ValueError) as e:
            raise SearchSpaceFromYamlFileError(e)
    """

    def __init__(self, exception):
        self.exception_type = type(exception).__name__
        self.message = (
            f"Error occurred during initialization of search space from "
            f"YAML file.\n {self.exception_type}: {exception}"
        )
        super().__init__(self.message)


def deduce_type(
    name: str, details: dict[str, str | int | float]
) -> str:
    """
    Deduces the parameter type from details and formats them.

    Args:
        name (str): The name of the parameter.
        details (dict): A dictionary containing parameter specifications.

    Returns:
        str: The deduced parameter type ('int', 'float', 'categorical', or 'constant').

    Raises:
        TypeError: If the type cannot be deduced or the details don't align with expected
        constraints.
    """
    try:
        # Deduce type
        if "type" in details:
            param_type = details.pop("type").lower()
        else:
            if isinstance(details, (str, int, float)):
                param_type = "const"
                return param_type
            else:
                param_type = deduce_param_type(name, details)
    except TypeError as e:
        if isinstance(details, (str, int, float)):
            param_type = "const"
            return param_type
        else:
            raise TypeError(
                f"Unable to deduce parameter type for '{name}' with details '{details}'.") \
                from e

    return param_type


def deduce_param_type(name: str, details: dict[str, int | str | float]) -> str:
    """Deduces the parameter type based on the provided details.

    The function interprets the 'details' dictionary to determine the parameter type.
    The dictionary should include key-value pairs that describe the parameter's
    characteristics, such as lower, upper, default value, or possible choices.


    Args:
        name (str): The name of the parameter.
        details ((dict[str, int | str | float])): A dictionary containing parameter
        specifications.

    Returns:
        str: The deduced parameter type ('int', 'float', 'categorical', or 'constant').

    Raises:
        TypeError: If the parameter type cannot be deduced from the details, or if the
        provided details have inconsistent types for expected keys.

    Example:
        param_type = deduce_param_type('example_param', {'lower': 0, 'upper': 10})"""
    # Logic to deduce type from details

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


def validate_integer_parameter(name: str, details: dict[str, str | int | float]):
    """
    Validates and processes an integer parameter's details, converting scientific
    notation to integers where necessary.

    This function checks the type of 'lower' and 'upper', and the 'default'
    value (if present) for an integer parameter. It also handles conversion of values
    in scientific notation (e.g., 1e2) to integers.

    Args:
        name (str): The name of the integer parameter.
        details (dict[str, str | int | float]): A dictionary containing the parameter's
                                                specifications. Expected keys include
                                                'lower', 'upper', and optionally 'default',
                                                among others.

    Raises:
        TypeError: If 'lower', 'upper', or 'default' are not valid integers or cannot
                   be converted from scientific notation to integers.
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
            # check if one value format is e notation and if its an integer
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
    if "default" in details:
        if not isinstance(details["default"], int):
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


def validate_float_parameter(name: str, details: dict[str, str | int | float]):
    """
    Validates and processes a float parameter's details, converting scientific
    notation values to float where necessary.

    This function checks the type of 'lower' and 'upper', and the 'default'
    value (if present) for a float parameter. It handles conversion of values in
    scientific notation (e.g., 1e-5) to float.

    Args:
        name: The name of the float parameter.
        details: A dictionary containing the parameter's specifications. Expected keys
                 include 'lower', 'upper', and optionally 'default', among others.

    Raises:
        TypeError: If 'lower', 'upper', or 'default' are not valid floats or cannot
                   be converted from scientific notation to floats.
    """
    # check if all keys are allowed to use and if the mandatory ones are provided

    if not isinstance(details["lower"], float) or not isinstance(details["upper"], float):
        try:
            # for numbers like 1e-5 and 10^
            details["lower"] = convert_scientific_notation(details["lower"])
            details["upper"] = convert_scientific_notation(details["upper"])
        except ValueError as e:
            raise TypeError(
                f"'lower' and 'upper' must be integer for " f"integer parameter '{name}'."
            ) from e
    if "default" in details:
        if not isinstance(details["default"], float):
            try:
                details["default"] = convert_scientific_notation(details["default"])
            except ValueError as e:
                raise TypeError(
                    f" default'{details['default']}' must be float for float "
                    f"parameter {name} "
                ) from e
    return details


def validate_categorical_parameter(name: str, details: dict[str, str | int | float]):
    """
    Validates a categorical parameter, including conversion of scientific notation
    values to floats within the choices.

    This function ensures that the 'choices' key in the details is a list and attempts
    to convert any elements in scientific notation to floats. It also handles the
    'default' value, converting it from scientific notation if necessary.

    Args:
        name: The name of the categorical parameter.
        details: A dictionary containing the parameter's specifications. Required key
                 is 'choices', with 'default' being optional.

    Raises:
        TypeError: If 'choices' is not a list
    """
    # check if all keys are allowed to use and if the mandatory ones are provided

    if not isinstance(details["choices"], list):
        raise TypeError(f"The 'choices' for '{name}' must be a list.")
    for i, element in enumerate(details["choices"]):
        try:
            converted_value, e_flag = convert_scientific_notation(
                element, show_usage_flag=True
            )
            if e_flag:
                details["choices"][
                    i
                ] = converted_value  # Replace the element at the same position
        except ValueError:
            pass  # If a ValueError occurs, simply continue to the next element
    if "default" in details:
        e_flag = False
        try:
            # check if e notation, if then convert to number
            default, e_flag = convert_scientific_notation(
                details["default"], show_usage_flag=True
            )
        except ValueError:
            pass  # if default value is not in a numeric format, Value Error occurs
        if e_flag is True:
            details["default"] = default
    return details


def formatting_constant(name: str, details: str | int | float):
    """
    Validates a constant parameter, including conversion of values in scientific
    notation to floats.

    This function checks the 'value' key in the details dictionary and converts any
    value expressed in scientific notation to a float. It ensures that the mandatory
    'value' key is provided and appropriately formatted.

    Args:
        name: The name of the constant parameter.
        details: A dictionary containing the parameter's specifications. The required
                 key is 'value'.
    """
    # check if all keys are allowed to use and if the mandatory ones are provided
    # check_keys(name, details, {"value", "type", "is_fidelity"}, {"value"})

    # check for e notation and convert it to float
    e_flag = False
    try:
        converted_value, e_flag = convert_scientific_notation(
            details, show_usage_flag=True
        )
    except ValueError:
        # if the value is not able to convert to float a ValueError get raised by
        # convert_scientific_notation function
        pass
    if e_flag:
        details = converted_value
    return details


