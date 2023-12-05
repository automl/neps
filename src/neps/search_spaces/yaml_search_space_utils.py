import re


def convert_scientific_notation(value, show_usage_flag=False):
    """Check if the value is a string that matches scientific e notation and convert it
    to float. (specially numbers like 3.3e-5 with a float value in front, which yaml
    can not interpret directly as float)."""

    e_notation_pattern = r"^-?\d+(\.\d+)?[eE]-?\d+$"

    flag = False  # Check if e notation was detected

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


def deduce_and_validate_param_type(name, details):
    """
    Deduces the parameter type from details and validates them.

    Args:
        name (str): The name of the parameter.
        details (dict): A dictionary containing parameter specifications.

    Returns:
        str: The deduced parameter type ('int', 'float', 'categorical', or 'constant').

    Raises:
        TypeError: If the type cannot be deduced or the details don't align with expected
        constraints.
    """
    # Deduce type
    if "type" in details:
        param_type = details["type"].lower()
    else:
        # Logic to infer type if not explicitly provided
        param_type = deduce_param_type(name, details)

    # Validate details based on deduced type
    validate_param_details(name, param_type, details)

    return param_type


def deduce_param_type(name, details):
    """Deduces the parameter type based on the provided details.

    This function analyzes the provided details dictionary to determine the type of
    parameter. It supports identifying integer, float, categorical, and constant
    parameter types.

    Args:
        name (str): The name of the parameter.
        details (dict): A dictionary containing parameter specifications.

    Returns:
        str: The deduced parameter type ('int', 'float', 'categorical', or 'constant').

    Raises:
        TypeError: If the parameter type cannot be deduced from the details, or if the
        provided details have inconsistent types for expected keys.

    Example:
        param_type = deduce_param_type('example_param', {'lower': 0, 'upper': 10})"""
    # Logic to deduce type from details
    if "lower" in details and "upper" in details:
        # Determine if it's an integer or float range parameter
        if isinstance(details["lower"], int) and isinstance(details["upper"], int):
            param_type = "int"
        elif isinstance(details["lower"], float) and isinstance(details["upper"], float):
            param_type = "float"
        else:
            details["lower"], flag_lower = convert_scientific_notation(
                details["lower"], show_usage_flag=True
            )
            details["upper"], flag_upper = convert_scientific_notation(
                details["upper"], show_usage_flag=True
            )
            # check if one value is 10^format to convert it to float
            if flag_lower or flag_upper:
                param_type = "float"
            else:
                raise TypeError(
                    f"Inconsistent types for 'lower' and 'upper' in '{name}'. "
                    f"Both must be either integers or floats."
                )

    elif "choices" in details:
        param_type = "categorical"
    elif "value" in details:
        param_type = "constant"
    else:
        raise TypeError(
            f"Unable to deduce parameter type from {name} "
            f"with details {details}\n"
            "Supported parameters:\n"
            "Float and Integer: Expected keys: 'lower', 'upper'\n"
            "Categorical: Expected keys: 'choices'\n"
            "Constant: Expected keys: 'value'"
        )
    return param_type


def validate_param_details(name, param_type, details):
    if not (isinstance(name, str) and isinstance(details, dict)):
        raise KeyError(
            f"Invalid format for {name} in YAML file. "
            f"Expected 'name' as string and corresponding 'details' as a "
            f"dictionary. Found 'name' type: {type(name).__name__}, 'details' "
            f"type: {type(details).__name__}."
        )
    param_type = param_type.lower()
    # init parameter by checking type
    if param_type in ("int", "integer"):
        # check if all keys are allowed
        check_allowed_keys(
            name,
            details,
            {
                "lower",
                "upper",
                "type",
                "log",
                "is_fidelity",
                "default",
                "default_confidence",
            },
        )
        # Check Integer Parameter
        if "lower" not in details or "upper" not in details:
            raise KeyError(
                f"Missing 'lower' or 'upper' for integer " f"parameter '{name}'."
            )
        if not isinstance(details["lower"], int) or not isinstance(details["upper"], int):
            try:
                # for numbers like 1e2 and 10^
                lower, flag_lower = convert_scientific_notation(
                    details["lower"], show_usage_flag=True
                )
                upper, flag_upper = convert_scientific_notation(
                    details["upper"], show_usage_flag=True
                )
                # check if one value format is e or 10^ and if its an integer
                if flag_lower or flag_upper:
                    if lower == int(lower) and upper == int(upper):
                        details["lower"] = int(lower)
                        details["upper"] = int(upper)
                    else:
                        raise ValueError()
                else:
                    raise ValueError()
            except ValueError as e:
                raise TypeError(
                    f"'lower' and 'upper' must be integer for "
                    f"integer parameter '{name}'."
                ) from e
        if "default" in details:
            if not isinstance(details["default"], int):
                default = convert_scientific_notation(details["default"])
                if default == int(default):
                    details["default"] = int(default)
                else:
                    raise TypeError(
                        f"default value {details['default']} "
                        f"must be integer for integer parameter {name}"
                    )

    elif param_type == "float":
        # check if all keys are allowed
        check_allowed_keys(
            name,
            details,
            {
                "lower",
                "upper",
                "type",
                "log",
                "is_fidelity",
                "default",
                "default_confidence",
            },
        )
        # Check Float Parameter
        if "lower" not in details or "upper" not in details:
            raise KeyError(
                f"Missing key 'lower' or 'upper' for float " f"parameter '{name}'."
            )
        if not isinstance(details["lower"], float) or not isinstance(
            details["upper"], float
        ):
            try:
                # for numbers like 1e-5 and 10^
                details["lower"] = convert_scientific_notation(details["lower"])
                details["upper"] = convert_scientific_notation(details["upper"])
            except ValueError as e:
                raise TypeError(
                    f"'lower' and 'upper' must be integer for "
                    f"integer parameter '{name}'."
                ) from e
        if "default" in details:
            if not isinstance(details["default"], float):
                try:
                    details["default"] = convert_scientific_notation(details["default"])
                except ValueError as e:
                    raise TypeError(
                        f" 'default' must be float for float parameter " f"{name} "
                    ) from e

    elif param_type in ("cat", "categorical"):
        # check if all keys are allowed
        check_allowed_keys(
            name,
            details,
            {"choices", "type", "is_fidelity", "default", "default_confidence"},
        )
        # Check Categorical parameter
        if "choices" not in details:
            raise KeyError(f"Missing key 'choices' for categorical " f"parameter {name}")
        if not isinstance(details["choices"], (list, tuple)):
            raise TypeError(f"The 'choices' for '{name}' must be a list or tuple.")
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
                pass
            if e_flag is True:
                details["default"] = default
    elif param_type in ("const", "constant"):
        # check if all keys are allowed
        check_allowed_keys(name, details, {"value", "type", "is_fidelity"})
        # Check Constant parameter
        if "value" not in details:
            raise KeyError(f"Missing key 'value' for constant parameter " f"{name}")
        else:
            e_flag = False
            try:
                converted_value, e_flag = convert_scientific_notation(
                    details["value"], show_usage_flag=True
                )
            except ValueError:
                pass
            if e_flag:
                details["value"] = converted_value

    else:
        # Handle unknown parameter types
        raise TypeError(
            f"Unsupported parameter type{details['type']} for '{name}'.\n"
            f"Supported Types for argument type are:\n"
            "For integer parameter: int, integer\n"
            "For float parameter: float\n"
            "For categorical parameter: cat, categorical\n"
            "For constant parameter: const, constant\n"
        )
    return param_type


def check_allowed_keys(name, my_dict, allowed_keys):
    """
    Checks if all keys in 'my_dict' are contained in the set 'allowed_keys'.
    If an unallowed key is found, an exception is raised.
    """
    for key in my_dict:
        if key not in allowed_keys:
            raise KeyError(f"This key is not allowed: '{key}' for parameter '{name}'")
