"""Float hyperparameter for search spaces."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Literal
from typing_extensions import Self, override

import numpy as np

from neps.search_spaces.domain import Domain
from neps.search_spaces.hyperparameters.numerical import Numerical

if TYPE_CHECKING:
    from neps.utils.types import Number


class Float(Numerical[float]):
    """A float value for a parameter.

    This kind of [`Parameter`][neps.search_spaces.parameter] is used
    to represent hyperparameters with continuous float values, optionally specifying if
    it exists
    on a log scale.
    For example, `l2_norm` could be a value in `(0.1)`, while the `learning_rate`
    hyperparameter in a neural network search space can be a `Float`
    with a range of `(0.0001, 0.1)` but on a log scale.

    ```python
    import neps

    l2_norm = neps.Float(0, 1)
    learning_rate = neps.Float(1e-4, 1e-1, log=True)
    ```

    Please see the [`Numerical`][neps.search_spaces.numerical.Numerical]
    class for more details on the methods available for this class.
    """

    DEFAULT_CONFIDENCE_SCORES: ClassVar[Mapping[str, float]] = {
        "low": 0.5,
        "medium": 0.25,
        "high": 0.125,
    }

    def __init__(
        self,
        lower: Number,
        upper: Number,
        *,
        log: bool = False,
        is_fidelity: bool = False,
        prior: Number | None = None,
        prior_confidence: Literal["low", "medium", "high"] = "low",
    ):
        """Create a new `Float`.

        Args:
            lower: lower bound for the hyperparameter.
            upper: upper bound for the hyperparameter.
            log: whether the hyperparameter is on a log scale.
            is_fidelity: whether the hyperparameter is fidelity.
            prior: prior value for the hyperparameter.
            prior_confidence: confidence score for the prior value, used when
                condsidering prior based optimization..
        """
        super().__init__(
            lower=float(lower),
            upper=float(upper),
            log=log,
            prior=float(prior) if prior is not None else None,
            prior_confidence=prior_confidence,
            is_fidelity=is_fidelity,
            domain=Domain.floating(lower, upper, log=log),
        )

    @override
    def clone(self) -> Self:
        clone = self.__class__(
            lower=self.lower,
            upper=self.upper,
            log=self.log,
            is_fidelity=self.is_fidelity,
            prior=self.prior,
            prior_confidence=self.prior_confidence_choice,
        )
        if self.value is not None:
            clone.set_value(self.value)

        return clone

    @override
    def set_value(self, value: float | None) -> None:
        if value is None:
            self._value = None
            self.normalized_value = None
            return

        if not self.lower <= value <= self.upper:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"{cls_name} parameter: prior bounds error. Expected lower <= prior"
                f" <= upper, but got lower={self.lower}, value={value},"
                f" upper={self.upper}"
            )

        value = float(value)
        self._value = value
        self.normalized_value = self.value_to_normalized(value)

    @override
    def sample_value(self, *, user_priors: bool = False) -> float:
        if self.log:
            assert self.log_bounds is not None
            low, high = self.log_bounds
            prior = self.log_prior
        else:
            low, high, prior = self.lower, self.upper, self.prior

        if user_priors and self.has_prior:
            dist, std = self._get_truncnorm_prior_and_std()
            value = dist.rvs() * std + prior
        else:
            value = np.random.uniform(low=low, high=high)

        if self.log:
            value = math.exp(value)

        return float(min(self.upper, max(self.lower, value)))

    @override
    def value_to_normalized(self, value: float) -> float:
        if self.log:
            assert self.log_bounds is not None
            low, high = self.log_bounds
        else:
            low, high = self.lower, self.upper

        value = np.log(value) if self.log else value
        return float((value - low) / (high - low))

    @override
    def normalized_to_value(self, normalized_value: float) -> float:
        if self.log:
            assert self.log_bounds is not None
            low, high = self.log_bounds
        else:
            low, high = self.lower, self.upper

        normalized_value = normalized_value * (high - low) + low
        _value = np.exp(normalized_value) if self.log else normalized_value
        return float(_value)

    def __repr__(self) -> str:
        float_repr = f"{self.value:.07f}" if self.value is not None else "None"
        return f"<Float, range: [{self.lower}, {self.upper}], value: {float_repr}>"


class FloatParameter(Float):
    """Deprecated: Use `Float` instead of `FloatParameter`.

    This class remains for backward compatibility and will raise a deprecation
    warning if used.
    """

    def __init__(
        self,
        lower: Number,
        upper: Number,
        *,
        log: bool = False,
        is_fidelity: bool = False,
        prior: Number | None = None,
        prior_confidence: Literal["low", "medium", "high"] = "low",
    ):
        """Initialize a deprecated `FloatParameter`.

        Args:
            lower: lower bound for the hyperparameter.
            upper: upper bound for the hyperparameter.
            log: whether the hyperparameter is on a log scale.
            is_fidelity: whether the hyperparameter is fidelity.
            prior: prior value for the hyperparameter.
            prior_confidence: confidence score for the prior value, used when
                condsidering prior based optimization..

        Raises:
            DeprecationWarning: A warning indicating that `neps.FloatParameter` is
            deprecated and `neps.Float` should be used instead.
        """
        import warnings

        warnings.warn(
            (
                "Usage of 'neps.FloatParameter' is deprecated and will be removed in"
                " future releases. Please use 'neps.Float' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            lower=lower,
            upper=upper,
            log=log,
            is_fidelity=is_fidelity,
            prior=prior,
            prior_confidence=prior_confidence,
        )
