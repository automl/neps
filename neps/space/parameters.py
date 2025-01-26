"""A module of all the parameters for the search space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
from more_itertools import all_unique

from neps.space.domain import Domain


@dataclass
class Float:
    """A float value for a parameter.

    This kind of parameter is used to represent hyperparameters with continuous float
    values, optionally specifying if it exists on a log scale.

    For example, `l2_norm` could be a value in `(0.1)`, while the `learning_rate`
    hyperparameter in a neural network search space can be a `Float`
    with a range of `(0.0001, 0.1)` but on a log scale.

    ```python
    import neps

    l2_norm = neps.Float(0, 1)
    learning_rate = neps.Float(1e-4, 1e-1, log=True)
    ```
    """

    lower: float
    """The lower bound of the numerical hyperparameter."""

    upper: float
    """The upper bound of the numerical hyperparameter."""

    center: float = field(init=False)
    """The center value of the numerical hyperparameter."""

    log: bool = False
    """Whether the hyperparameter is in log space."""

    prior: float | None = None
    """Prior value for the hyperparameter."""

    prior_confidence: Literal["low", "medium", "high"] = "low"
    """Confidence score for the prior value when considering prior based optimization."""

    is_fidelity: bool = False
    """Whether the hyperparameter is fidelity."""

    domain: Domain[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError(
                f"Float parameter: bounds error (lower >= upper). Actual values: "
                f"lower={self.lower}, upper={self.upper}"
            )

        if self.log and (self.lower <= 0 or self.upper <= 0):
            raise ValueError(
                f"Float parameter: bounds error (log scale cant have bounds <= 0). "
                f"Actual values: lower={self.lower}, upper={self.upper}"
            )

        if self.prior is not None and not self.lower <= self.prior <= self.upper:
            raise ValueError(
                f"Float parameter: prior bounds error. Expected lower <= prior <= upper, "
                f"but got lower={self.lower}, prior={self.prior}, upper={self.upper}"
            )

        self.lower = float(self.lower)
        self.upper = float(self.upper)

        if self.is_fidelity and (self.lower < 0 or self.upper < 0):
            raise ValueError(
                f"Float parameter: fidelity bounds error. Expected fidelity"
                f" bounds to be >= 0, but got lower={self.lower}, "
                f" upper={self.upper}."
            )

        if np.isnan(self.lower):
            raise ValueError("Can not have lower bound that is nan")

        if np.isnan(self.upper):
            raise ValueError("Can not have upper bound that is nan")

        self.domain = Domain.floating(self.lower, self.upper, log=self.log)
        self.center = self.domain.cast_one(0.5, frm=Domain.unit_float())


@dataclass
class Integer:
    """An integer value for a parameter.

    This kind of parameter is used to represent hyperparameters with
    continuous integer values, optionally specifying f it exists on a log scale.

    For example, `batch_size` could be a value in `(32, 128)`, while the `num_layers`
    hyperparameter in a neural network search space can be a `Integer`
    with a range of `(1, 1000)` but on a log scale.

    ```python
    import neps

    batch_size = neps.Integer(32, 128)
    num_layers = neps.Integer(1, 1000, log=True)
    ```
    """

    lower: int
    """The lower bound of the numerical hyperparameter."""

    upper: int
    """The upper bound of the numerical hyperparameter."""

    center: int = field(init=False)
    """The center value of the numerical hyperparameter."""

    log: bool = False
    """Whether the hyperparameter is in log space."""

    prior: int | None = None
    """Prior value for the hyperparameter."""

    prior_confidence: Literal["low", "medium", "high"] = "low"
    """Confidence score for the prior value when considering prior based optimization."""

    is_fidelity: bool = False
    """Whether the hyperparameter is fidelity."""

    domain: Domain[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError(
                f"Integer parameter: bounds error (lower >= upper). Actual values: "
                f"lower={self.lower}, upper={self.upper}"
            )

        # We could get scientific notation such as 1e3 which would be a float.
        # However we can safely cast this to `int` so we do an equality check
        # to see if a possible float value matches its int value, before raising
        # about floats.
        lower_int = int(self.lower)
        upper_int = int(self.upper)
        if lower_int != self.lower or upper_int != self.upper:
            raise ValueError(
                f"Integer parameter: bounds error (lower and upper must be integers). "
                f"Actual values: lower={self.lower}, upper={self.upper}"
            )

        self.lower = lower_int
        self.upper = upper_int

        if self.is_fidelity and (self.lower < 0 or self.upper < 0):
            raise ValueError(
                f"Integer parameter: fidelity bounds error. Expected fidelity"
                f" bounds to be >= 0, but got lower={self.lower}, "
                f" upper={self.upper}."
            )

        if self.log and (self.lower <= 0 or self.upper <= 0):
            raise ValueError(
                f"Integer parameter: bounds error (log scale cant have bounds <= 0). "
                f"Actual values: lower={self.lower}, upper={self.upper}"
            )

        if self.prior is not None and not self.lower <= self.prior <= self.upper:
            raise ValueError(
                f"Integer parameter: Expected lower <= prior <= upper,"
                f"but got lower={self.lower}, prior={self.prior}, upper={self.upper}"
            )

        self.domain = Domain.integer(self.lower, self.upper, log=self.log)
        self.center = self.domain.cast_one(0.5, frm=Domain.unit_float())


@dataclass
class Categorical:
    """A list of **unordered** choices for a parameter.

    This kind of parameter is used to represent hyperparameters that can take on a
    discrete set of unordered values. For example, the `optimizer` hyperparameter
    in a neural network search space can be a `Categorical` with choices like
    `#!python ["adam", "sgd", "rmsprop"]`.

    ```python
    import neps

    optimizer_choice = neps.Categorical(
        ["adam", "sgd", "rmsprop"],
        prior="adam"
    )
    ```
    """

    choices: list[float | int | str]
    """The list of choices for the categorical hyperparameter."""

    prior: float | int | str | None = None
    """The default value for the categorical hyperparameter."""

    center: float | int | str = field(init=False)
    """The center value of the categorical hyperparameter.

    As there is no natural center for a categorical parameter, this is the
    first value in the choices list.
    """

    prior_confidence: Literal["low", "medium", "high"] = "low"
    """Confidence score for the prior value when considering prior based optimization."""

    domain: Domain[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.choices = list(self.choices)

        if len(self.choices) <= 1:
            raise ValueError("Categorical choices must have more than one value.")

        for choice in self.choices:
            if not isinstance(choice, float | int | str):
                raise TypeError(
                    f'Choice "{choice}" is not of a valid type (float, int, str)'
                )

        if not all_unique(self.choices):
            raise ValueError(f"Choices must be unique, got duplicates.\n{self.choices}")

        if self.prior is not None and self.prior not in self.choices:
            raise ValueError(
                f"Default value {self.prior} is not in the provided"
                f" choices {self.choices}"
            )

        self.domain = Domain.indices(len(self.choices))
        self.center = self.choices[0]


@dataclass
class Constant:
    """A constant value for a parameter.

    This kind of parameter is used to represent hyperparameters with values that
    should not change during optimization.

    For example, the `batch_size` hyperparameter in a neural network search space
    can be a `Constant` with a value of `32`.

    ```python
    import neps

    batch_size = neps.Constant(32)
    ```
    """

    value: Any


Parameter: TypeAlias = Float | Integer | Categorical
"""A type alias for all the parameter types.

* [`Float`][neps.space.Float]
* [`Integer`][neps.space.Integer]
* [`Categorical`][neps.space.Categorical]

A [`Constant`][neps.space.Constant] is not included as it does not change value.
"""
