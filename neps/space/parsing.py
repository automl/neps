"""This module contains functions for parsing search spaces."""

from __future__ import annotations

import dataclasses
import logging
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

from neps.space.parameters import Categorical, Constant, Float, Integer, Parameter
from neps.space.new_space.space import Pipeline
from neps.space.search_space import SearchSpace

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

logger = logging.getLogger("neps")

E_NOTATION_PATTERN = r"^-?\d+(\.\d+)?[eE]-?\d+$"


def scientific_parse(value: str | int | float) -> str | int | float:
    """Parse a value that may be scientific notation."""
    if not isinstance(value, str):
        return value

    value_no_space = value.replace(" ", "")
    is_scientific = re.match(E_NOTATION_PATTERN, value_no_space)

    if not is_scientific:
        return value

    # We know there's an 'e' in the string,
    # Now we need to check if its an integer or float
    # `int` wont parse scientific notation so we first cast to float
    # and see if it's the same as the int cast
    float_val = float(value_no_space)
    int_val = int(float_val)
    if float_val == int_val:
        return int_val

    return float_val


SerializedParameter: TypeAlias = (
    Mapping[str, Any]  # {"type": "int", ...}
    | str  # const
    | int  # const
    | float  # const
    | tuple[int, int]  # int
    | tuple[float, float]  # float
    | tuple[int | float | str, int | float | str]  # bounds (with scientific not.)
    | list[int | str | float]  # categorical
)


def as_parameter(details: SerializedParameter) -> Parameter | Constant:  # noqa: C901, PLR0911, PLR0912
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
        # Constant
        case str() | int() | float():
            val = scientific_parse(details)
            return Constant(val)

        # Bounds of float or int
        case tuple((x, y)):
            _x = scientific_parse(x)
            _y = scientific_parse(y)
            match (_x, _y):
                case (int(), int()):
                    return Integer(_x, _y)
                case (float(), float()):
                    return Float(_x, _y)
                case _:
                    raise ValueError(
                        f"Expected both 'int' or 'float' for bounds but got {type(_x)=}"
                        f" and {type(_y)=}."
                    )
        # Matches any sequence of length 2. We could have the issue that the user
        # deserializes a yaml tuple pair which gets converted to a list.
        # We interpret this as bounds if:
        # 1. There are 2 elements
        # 2. Both elements are co-ercible to the same number type
        # 3. They are ordered
        case (x, y):  # 1.
            _x = scientific_parse(x)
            _y = scientific_parse(y)
            match (_x, _y):
                case (int(), int()) if _x <= _y:  # 2./3.
                    return Integer(_x, _y)
                case (float(), float()) if _x <= _y:  # 2./3.
                    return Float(_x, _y)

                # Error case:
                # We do have two numbers, but of different types. This could
                # be user error so rather than guess, we raise an error.
                case (int(), float()) | (float(), int()):
                    raise ValueError(
                        f"Got a mix of a float and an int with {details=},"
                        " tried to interpret these as bounds but found them to be"
                        " different types."
                        "\nIf you wanted to specify a categorical, i.e. a discrete"
                        f" choices between the values {x=} and {y=}, then you can use"
                        " the more verbose syntax of specifying 'type: cat'."
                        "\nIf you did intend to specify bounds, then ensure that"
                        " the values are both of the same type."
                    )
                # At least one of them is a string, so we treat is as categorical.
                case _:
                    return Categorical(choices=[_x, _y])

        ## Categorical list of choices (tuple is reserved for bounds)
        case Sequence() if not isinstance(details, tuple):
            # It's unlikely that if we find an element that can be converted to
            # scientific notation that we wouldn't want to do so, for example,
            # when specifying a grid. Hence, we map over the list and convert
            # what we can
            details = [scientific_parse(d) for d in details]
            return Categorical(details)

        # Categorical dict declartion
        case {"choices": choices, **rest}:
            _type = rest.pop("type", None)
            if _type is not None and _type not in ("cat", "categorical"):
                raise ValueError(f"Unrecognized type '{_type}' with 'choices' set.")

            # See note above about scientific notation elements
            choices = [scientific_parse(c) for c in choices]
            return Categorical(choices, **rest)  # type: ignore

        # Constant dict declartion
        case {"value": v, **_rest}:
            _type = _rest.pop("type", None)
            if _type is not None and _type not in ("const", "constant"):
                raise ValueError(
                    f"Unrecognized type '{_type}' with 'value' set,"
                    f" which indicates to treat value `{v}` a constant."
                )

            return Constant(v, **_rest)  # type: ignore

        # Bounds dict declartion
        case {"lower": l, "upper": u, **rest}:
            _x = scientific_parse(l)
            _y = scientific_parse(u)

            _type = rest.pop("type", None)
            match _type:
                case "int" | "integer":
                    return Integer(_x, _y, **rest)  # type: ignore
                case "float" | "floating":
                    return Float(_x, _y, **rest)  # type: ignore
                case None:
                    match (_x, _y):
                        case (int(), int()):
                            return Integer(_x, _y, **rest)  # type: ignore
                        case (float(), float()):
                            return Float(_x, _y, **rest)  # type: ignore
                        case _:
                            raise ValueError(
                                f"Expected both 'int' or 'float' for bounds but"
                                f" got {type(_x)=} and {type(_y)=}."
                            )
                case _:
                    raise ValueError(
                        f"Unrecognized type '{_type}' with both a 'lower'"
                        " and 'upper' set."
                    )
        case _:
            raise ValueError(
                f"Unable to deduce parameter with details '{details}'."
                " Please see our documentation for details."
            )


def convert_mapping(pipeline_space: Mapping[str, Any]) -> SearchSpace:
    """Converts a dictionary to a SearchSpace object."""
    parameters: dict[str, Parameter | Constant] = {}
    for name, details in pipeline_space.items():
        match details:
            case Float() | Integer() | Categorical() | Constant():
                parameters[name] = dataclasses.replace(details)  # copy
            case str() | int() | float() | Mapping():
                try:
                    parameters[name] = as_parameter(details)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Error parsing parameter '{name}'") from e
            case None:
                parameters[name] = Constant(None)
            case _:
                raise ValueError(
                    f"Unrecognized parameter type '{type(details)}' for '{name}'."
                )

    return SearchSpace(parameters)


def convert_configspace(configspace: ConfigurationSpace) -> SearchSpace:
    """Constructs a [`SearchSpace`][neps.space.SearchSpace]
    from a [`ConfigurationSpace`](https://automl.github.io/ConfigSpace/latest/).

    Args:
        configspace: The configuration space to construct the pipeline space from.

    Returns:
        A dictionary where keys are parameter names and values are parameter objects.
    """
    import ConfigSpace as CS

    space: dict[str, Parameter | Constant] = {}
    if any(configspace.conditions) or any(configspace.forbidden_clauses):
        raise NotImplementedError(
            "The ConfigurationSpace has conditions or forbidden clauses, "
            "which are not supported by neps."
        )

    for name, hyperparameter in configspace.items():
        match hyperparameter:
            case CS.Constant():
                space[name] = Constant(value=hyperparameter.value)
            case CS.CategoricalHyperparameter():
                space[name] = Categorical(hyperparameter.choices)  # type: ignore
            case CS.OrdinalHyperparameter():
                raise ValueError(
                    "NePS does not support ordinals yet, please"
                    " either convert it to an integer or use a"
                    " categorical hyperparameter."
                )
            case CS.UniformIntegerHyperparameter():
                space[name] = Integer(
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    log=hyperparameter.log,
                    prior=None,
                )
            case CS.UniformFloatHyperparameter():
                space[name] = Float(
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    log=hyperparameter.log,
                    prior=None,
                )

            case CS.NormalFloatHyperparameter():
                warnings.warn(
                    "NormalFloatHyperparameter is detected as a prior for NePS"
                    " and will will consider it as a 'medium' prior_confidence."
                    " If you wish to silence this warning, please manually"
                    " convert your ConfigurationSpace to a SearchSpace.",
                    UserWarning,
                    stacklevel=2,
                )
                space[name] = Float(
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    log=hyperparameter.log,
                    prior=hyperparameter.mu,
                )
            case CS.NormalIntegerHyperparameter():
                warnings.warn(
                    "NormalIntegerHyperparameter is detected as a prior for NePS"
                    " and will will consider it as a 'medium' prior_confidence."
                    " If you wish to silence this warning, please manually"
                    " convert your ConfigurationSpace to a SearchSpace.",
                    UserWarning,
                    stacklevel=2,
                )
                space[name] = Integer(
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    log=hyperparameter.log,
                    prior=int(hyperparameter.mu),
                )
            case _:
                raise ValueError(f"Unknown hyperparameter type {hyperparameter}")

    return SearchSpace(space)


def convert_to_space(
    space: (
        Mapping[str, dict | str | int | float | Parameter]
        | SearchSpace
        | ConfigurationSpace
        | Pipeline
    ),
) -> SearchSpace | Pipeline:
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
        case Pipeline():
            return space
        case _:
            raise ValueError(
                f"Unsupported type '{type(space)}' for conversion to SearchSpace."
            )
