"""This module contains functions for parsing search spaces."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, overload

from neps.search_spaces import Categorical, Constant, Float, Integer, Parameter
from neps.search_spaces.search_space import SearchSpace

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

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
    """Convert a value to a float if it's a string that matches scientific enotation.

    This is especially useful for numbers like "3.3e-5" which YAML parsers may not
    directly interpret as floats.

    If the 'show_usage_flag' is set to True, the function returns a tuple of the float
    conversion and a boolean flag indicating whether scientific notation was detected.

    Args:
        value: The value to convert. Can be an integer, float,
          or a string representing a number, possibly in scientific notation.
        show_usage_flag: If True, the function also returns
            a flag indicating whether scientific notation was detected
            in the string.

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


def formatting_int(details: dict[str, str | int | float]) -> dict:
    """Converts scientific notation values to integers.

    This function converts the 'lower' and 'upper' bounds, as well as the 'default'
    value (if present), from scientific notation to integers.

    Args:
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
            raise TypeError("'lower' and 'upper' must be integer") from e

    if "default" in details and not isinstance(details["default"], int):
        try:
            # convert value can raise ValueError
            default = convert_scientific_notation(details["default"])
            if default == int(default):
                details["default"] = int(default)
            else:
                raise TypeError()  # type of value is not int
        except (ValueError, TypeError) as e:
            raise TypeError(f"default value {details['default']} must be int") from e

    return details


def formatting_float(details: dict[str, str | int | float]) -> dict:
    """Converts scientific notation values to floats.

    This function converts the 'lower' and 'upper' bounds, as well as the 'default'
    value (if present), from scientific notation to floats.

    Args:
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
            raise TypeError("'lower' and 'upper' must be float") from e

    if "default" in details and not isinstance(details["default"], float):
        try:
            details["default"] = convert_scientific_notation(details["default"])
        except ValueError as e:
            raise TypeError(f" default'{details['default']}' must be float") from e

    return details


def formatting_cat(details: dict[str, list | str | int | float]) -> dict:
    """This function ensures that the 'choices' key in the details is a list and attempts
    to convert any elements expressed in scientific notation to floats. It also handles
    the 'default' value, converting it from scientific notation if necessary.

    Args:
        details: A dictionary containing the parameter's specifications. The required key
                 is 'choices', which must be a list. The 'default' key is optional.

    Raises:
        TypeError: If 'choices' is not a list.

    Returns:
        The validated and possibly converted categorical parameter details.
    """
    if not isinstance(details["choices"], list):
        raise TypeError("The 'choices' for must be a list.")

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
        if not isinstance(extracted_default, str | int | float):
            raise TypeError(
                f"The 'default' value must be a string, integer, or float."
                f" Got {type(extracted_default).__name__}."
            )

        try:  # noqa: SIM105
            # check if e notation, if then convert to number
            default, e_flag = convert_scientific_notation(
                extracted_default, show_usage_flag=True
            )
        except ValueError:
            pass  # if default value is not in a numeric format, Value Error occurs

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
    # check for e notation and convert it to float
    e_flag = False
    try:  # noqa: SIM105
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


def deduce_type(  # noqa: PLR0911
    details: Mapping[str, str | int | float] | str | int | float,
) -> Literal["int", "float", "categorical", "const"]:
    """Deduces the parameter type from details.

    Args:
        details: A dictionary containing parameter specifications or
            a direct value (string, integer, or float).

    Returns:
        The deduced parameter type ('int', 'float', 'categorical', or 'constant').

    Raises:
        TypeError: If the type cannot be deduced or the details don't align with expected
                constraints.
    """
    match details:
        case str() | int() | float():
            return "const"
        case {"type": v, **_rest} if isinstance(v, str):
            return v.lower()  # type: ignore
        case {"choices": _}:
            return "categorical"
        case {"lower": l, "upper": u} if isinstance(l, int) and isinstance(u, int):
            _type = details.get("type")
            if _type not in ("int", "integer", None):
                raise ValueError(
                    f"Expected type 'int' or 'integer' for details '{details}'"
                    " when 'lower' and 'upper' are integers."
                )
            return "int"
        case {"lower": l, "upper": u} if isinstance(l, float) and isinstance(u, float):
            _type = details.get("type")
            if _type not in ("float", None):
                raise ValueError(
                    f"Expected type 'float' for details '{details}'"
                    " when 'lower' and 'upper' are floats."
                )
            return "float"
        case {"lower": l, "upper": u}:
            lower = convert_scientific_notation(l) if isinstance(l, str) else l
            upper = convert_scientific_notation(u) if isinstance(u, str) else u

            if isinstance(lower, int) and isinstance(upper, int):
                return "int"

            if isinstance(lower, float) and isinstance(upper, float):
                return "float"

            raise TypeError(
                f"Unable to deduce parameter with details {details}."
                " Expected 'lower' and 'upper' to both be integers or floats"
                f" but got {type(lower)=} and {type(upper)=}."
            )
        case _:
            raise TypeError(f"Unable to deduce parameter with details '{details}'.")


def as_parameter(config: Mapping[str, Any] | str | int | float) -> Parameter:
    """Converts a configuration dictionary to a Parameter object."""
    _type = deduce_type(config)
    if isinstance(config, str | int | float):
        assert _type == "const"
        value = convert_scientific_notation(config) if isinstance(config, str) else config
        return Constant(value)  # type: ignore

    match _type:
        case "int" | "integer":
            formatted_details = formatting_int(dict(config))
            formatted_details.pop("type", None)
            return Integer(**formatted_details)
        case "float":
            formatted_details = formatting_float(dict(config))
            formatted_details.pop("type", None)
            return Float(**formatted_details)
        case "cat" | "categorical":
            formatted_details = formatting_cat(dict(config))
            formatted_details.pop("type", None)
            return Categorical(**formatted_details)
        case _:
            raise ValueError(
                f"Unrecognized type '{_type}'. Valid options"
                " are 'int', 'float', 'categorical', 'const'."
            )


def convert_mapping(pipeline_space: Mapping[str, Any]) -> SearchSpace:
    """Converts a dictionary to a SearchSpace object."""
    parameters: dict[str, Parameter] = {}
    for name, details in pipeline_space.items():
        match details:
            case Parameter():
                parameters[name] = details.clone()
            case str() | int() | float() | Mapping():
                try:
                    parameters[name] = as_parameter(details)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Error parsing parameter '{name}'") from e
            case _:
                raise ValueError(
                    f"Unrecognized parameter type '{type(details)}' for '{name}'."
                )

    return SearchSpace(**parameters)


def convert_configspace(
    configspace: ConfigurationSpace,
) -> SearchSpace:
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
            parameter = Constant(value=hyperparameter.value)
        elif isinstance(hyperparameter, CS.CategoricalHyperparameter):
            parameter = Categorical(
                hyperparameter.choices,
                prior=hyperparameter.default_value,
            )
        elif isinstance(hyperparameter, CS.OrdinalHyperparameter):
            parameter = Categorical(
                hyperparameter.sequence,
                prior=hyperparameter.default_value,
            )
        elif isinstance(hyperparameter, CS.UniformIntegerHyperparameter):
            parameter = Integer(
                lower=hyperparameter.lower,
                upper=hyperparameter.upper,
                log=hyperparameter.log,
                prior=hyperparameter.default_value,
            )
        elif isinstance(hyperparameter, CS.UniformFloatHyperparameter):
            parameter = Float(
                lower=hyperparameter.lower,
                upper=hyperparameter.upper,
                log=hyperparameter.log,
                prior=hyperparameter.default_value,
            )
        else:
            raise ValueError(f"Unknown hyperparameter type {hyperparameter}")
        pipeline_space[hyperparameter.name] = parameter
    return SearchSpace(**pipeline_space)


def convert_to_space(
    space: (
        Mapping[str, dict | str | int | float | Parameter]
        | SearchSpace
        | ConfigurationSpace
    ),
) -> SearchSpace:
    """Converts a search space to a SearchSpace object.

    Args:
        space: The search space to convert.

    Returns:
        The SearchSpace object representing the search space.
    """
    # We quickly check ConfigSpace becuse it inherits from Mapping
    try:
        from ConfigSpace import ConfigurationSpace

        if isinstance(space, ConfigurationSpace):
            return convert_configspace(space)
    except ImportError:
        pass

    match space:
        case SearchSpace():
            return space
        case Mapping():
            return convert_mapping(space)
        case _:
            raise ValueError(
                f"Unsupported type '{type(space)}' for conversion to SearchSpace."
            )
