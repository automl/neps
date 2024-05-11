"""Float hyperparameter for search spaces."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, Literal, Mapping
from typing_extensions import override

import numpy as np

from neps.search_spaces.hyperparameters.numerical import NumericalParameter

if TYPE_CHECKING:
    from neps.types import Number


class FloatParameter(NumericalParameter[float]):
    """A float value for a parameter.

    This kind of [`Parameter`][neps.search_spaces.parameter] is used
    to represent hyperparameters with continuous float values, optionally specifying if
    it exists
    on a log scale.
    For example, `l2_norm` could be a value in `(0.1)`, while the `learning_rate`
    hyperparameter in a neural network search space can be a `FloatParameter`
    with a range of `(0.0001, 0.1)` but on a log scale.

    ```python
    import neps

    l2_norm = neps.FloatParameter(0, 1)
    learning_rate = neps.FloatParameter(1e-4, 1e-1, log=True)
    ```

    Please see the [`NumericalParameter`][neps.search_spaces.numerical.NumericalParameter]
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
        default: Number | None = None,
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        """Create a new `FloatParameter`.

        Args:
            lower: lower bound for the hyperparameter.
            upper: upper bound for the hyperparameter.
            log: whether the hyperparameter is on a log scale.
            is_fidelity: whether the hyperparameter is fidelity.
            default: default value for the hyperparameter.
            default_confidence: confidence score for the default value, used when
                condsidering prior based optimization..
        """
        super().__init__(
            lower=float(lower),
            upper=float(upper),
            log=log,
            default=float(default) if default is not None else None,
            default_confidence=default_confidence,
            is_fidelity=is_fidelity,
        )

    @override
    def set_default(self, default: float | None) -> None:
        if default is None:
            self.default = None
            self.has_prior = False
            self.log_default = None
            return

        if not self.lower <= default <= self.upper:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"{cls_name} parameter: default bounds error. Expected lower <= default"
                f" <= upper, but got lower={self.lower}, default={default},"
                f" upper={self.upper}"
            )

        self.default = float(default)
        self.has_prior = default is not None
        if self.log:
            self.log_default = np.log(self.default)

    @override
    def set_value(self, value: float | None) -> None:
        if value is None:
            self._value = None
            self.normalized_value = None
            self.log_value = None
            return

        if not self.lower <= value <= self.upper:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"{cls_name} parameter: default bounds error. Expected lower <= default"
                f" <= upper, but got lower={self.lower}, value={value},"
                f" upper={self.upper}"
            )

        value = float(value)
        self._value = value
        self.normalized_value = self.value_to_normalized(value)
        if self.log:
            self.log_value = np.log(value)

    @override
    def sample_value(self, *, user_priors: bool = False) -> float:
        if self.log:
            assert self.log_bounds is not None
            low, high = self.log_bounds
            default = self.log_default
        else:
            low, high, default = self.lower, self.upper, self.default

        if user_priors and self.has_prior:
            dist, std = self._get_truncnorm_prior_and_std()
            value = dist.rvs() * std + default
        else:
            value = np.random.uniform(low=low, high=high)

        if self.log:
            value = math.exp(value)

        return float(min(self.upper, max(self.lower, value)))

    @override
    def value_to_normalized(self, value: float) -> float:
        if np.isnan(value):
            raise ValueError("Float parameter value is NaN!")

        if self.log:
            assert self.log_bounds is not None
            low, high = self.log_bounds
        else:
            low, high = self.lower, self.upper

        value = np.log(value) if self.log else value
        return float((value - low) / (high - low))

    @override
    def normalized_to_value(self, normalized_value: float) -> float:
        if np.isnan(normalized_value):
            raise ValueError("Float parameter value is NaN!")

        if self.log:
            assert self.log_bounds is not None
            low, high = self.log_bounds
        else:
            low, high = self.lower, self.upper

        normalized_value = normalized_value * (high - low) + low
        return np.exp(normalized_value) if self.log else normalized_value

    def __repr__(self) -> str:
        float_repr = f"{self.value:.07f}" if self.value is not None else "None"
        return f"<Float, range: [{self.lower}, {self.upper}], value: {float_repr}>"
