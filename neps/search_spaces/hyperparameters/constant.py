from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T", int, float, str)


@dataclass
class ConstantParameter(Generic[T]):
    """A constant value for a parameter.

    This kind of [`Parameter`][neps.search_spaces.parameter] is used
    to represent hyperparameters with a fixed value. For example, the
    `num_classes` hyperparameter in a neural network search space can be a
    `ConstantParameter` with a value of `10`.

    ```python
    import neps

    num_classes = neps.ConstantParameter(10)
    ```
    """

    value: T
    """The constant value for the parameter."""
