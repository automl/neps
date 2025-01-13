"""The base [`Parameter`][neps.search_spaces.Parameter] class.

The `Parameter` refers to both the hyperparameter definition but also
holds a [`.value`][neps.search_spaces.Parameter.value] which can be
set or empty, in which case it is `None`.

!!! tip

    A `Parameter` which allows for defining a
    [`.default`][neps.search_spaces.Parameter.default] and some prior,
    i.e. some default value along with a confidence that this is a good setting,
    should implement the [`ParameterWithPrior`][neps.search_spaces.ParameterWithPrior]
    class.

    This is utilized by certain optimization routines to inform the search process.
"""

from __future__ import annotations

from abc import ABC
from typing import Generic, TypeAlias, TypeVar

class Parameter(ABC):
    """A base class for hyperparameters.

    Attributes:
        prior: default value for the hyperparameter. This value
            is used as a prior to inform algorithms about a decent
            default value for the hyperparameter, as well as use
            attributes from [`ParameterWithPrior`][neps.search_spaces.ParameterWithPrior],
            to aid in optimization.
        is_fidelity: whether the hyperparameter is fidelity.
        value: value for the hyperparameter, if any.
        normalized_value: normalized value for the hyperparameter.
    """



class ParameterWithPrior(Parameter[ValueT, SerializedT]):
    """A base class for hyperparameters with priors.

    Attributes:
        prior_confidence_choice: The choice of how confident any algorithm should
            be in the prior value being a good value.
        prior_confidence_score: A score used by algorithms to utilize the prior value.
        has_prior: whether the hyperparameter has a prior that can be used by an
            algorithm. In many cases, this refers to having a prior value.
    """

    prior_confidence_choice: str
    prior_confidence_score: float
    has_prior: bool
