from typing_extensions import TypeAlias

from neps.search_spaces.hyperparameters.categorical import CategoricalParameter
from neps.search_spaces.hyperparameters.constant import ConstantParameter
from neps.search_spaces.hyperparameters.numerical import (
    FloatParameter,
    IntegerParameter,
    NumericalParameter,
)

Parameter: TypeAlias = (
    CategoricalParameter
    | NumericalParameter
    | FloatParameter
    | IntegerParameter
    | ConstantParameter
)
"""Type alias for all hyperparameter types.

Note:
    As we don't expect to add more hyperparameter types, nor do we share enough
    implementation details or how they are used between optimizers, we make
    this a discrete type union and not a base class.

    This may change.
"""

__all__ = [
    "CategoricalParameter",
    "ConstantParameter",
    "FloatParameter",
    "IntegerParameter",
    "NumericalParameter",
    "Parameter",
]
