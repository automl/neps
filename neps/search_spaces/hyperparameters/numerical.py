"""The [`Numerical`][neps.search_spaces.Numerical] is
a [`Parameter`][neps.search_spaces.Parameter] that represents a numerical
range.

The two primary numerical hyperparameters are:

* [`Float`][neps.search_spaces.Float] for continuous
    float values.
* [`Integer`][neps.search_spaces.Integer] for discrete
    integer values.

The [`Numerical`][neps.search_spaces.Numerical] is a
base class for both of these hyperparameters, and includes methods from
both [`ParameterWithPrior`][neps.search_spaces.ParameterWithPrior],
allowing you to set a confidence along with a
[`.default`][neps.search_spaces.Parameter.default] that can be used
with certain algorithms.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar
from typing_extensions import override

import numpy as np
import scipy

from neps.search_spaces.parameter import ParameterWithPrior

if TYPE_CHECKING:
    from neps.search_spaces.domain import Domain
    from neps.utils.types import TruncNorm

T = TypeVar("T", int, float)


# OPTIM(eddiebergman): When calculating priors over and over,
# creating this scipy.rvs is surprisingly slow. Since we do not
# mutate them, we just cache them. This is done across instances so
# we also can access this cache with new copies of the hyperparameters.
@lru_cache(maxsize=128, typed=False)
def _get_truncnorm_prior_and_std(
    low: int | float,
    high: int | float,
    default: int | float,
    confidence_score: float,
) -> tuple[TruncNorm, float]:
    std = (high - low) * confidence_score
    a, b = (low - default) / std, (high - default) / std
    return scipy.stats.truncnorm(a, b), float(std)


class Numerical(ParameterWithPrior[T, T]):
    """A numerical hyperparameter is bounded by a lower and upper value.

    Attributes:
        lower: The lower bound of the numerical hyperparameter.
        upper: The upper bound of the numerical hyperparameter.
        log: Whether the hyperparameter is in log space.
        log_bounds: The log bounds of the hyperparameter, if `log=True`.
        log_default: The log default value of the hyperparameter, if `log=True`
            and a `default` is set.
        default_confidence_choice: The default confidence choice.
        default_confidence_score: The default confidence score.
        has_prior: Whether the hyperparameter has a prior.
    """

    DEFAULT_CONFIDENCE_SCORES: ClassVar[Mapping[str, float]]

    def __init__(
        self,
        lower: T,
        upper: T,
        *,
        log: bool = False,
        default: T | None,
        is_fidelity: bool,
        domain: Domain[T],
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        """Initialize the numerical hyperparameter.

        Args:
            lower: The lower bound of the numerical hyperparameter.
            upper: The upper bound of the numerical hyperparameter.
            log: Whether the hyperparameter is in log space.
            default: The default value of the hyperparameter.
            is_fidelity: Whether the hyperparameter is a fidelity parameter.
            domain: The domain of the hyperparameter.
            default_confidence: The default confidence choice.
        """
        super().__init__(value=None, default=default, is_fidelity=is_fidelity)  # type: ignore
        _cls_name = self.__class__.__name__
        if lower >= upper:
            raise ValueError(
                f"{_cls_name} parameter: bounds error (lower >= upper). Actual values: "
                f"lower={lower}, upper={upper}"
            )

        if log and (lower <= 0 or upper <= 0):
            raise ValueError(
                f"{_cls_name} parameter: bounds error (log scale cant have bounds <= 0)."
                f" Actual values: lower={lower}, upper={upper}"
            )

        if default is not None and not lower <= default <= upper:
            raise ValueError(
                f"Float parameter: default bounds error. Expected lower <= default"
                f" <= upper, but got lower={lower}, default={default},"
                f" upper={upper}"
            )

        if default_confidence not in self.DEFAULT_CONFIDENCE_SCORES:
            raise ValueError(
                f"{_cls_name} parameter: default confidence score error. Expected one of "
                f"{list(self.DEFAULT_CONFIDENCE_SCORES.keys())}, but got "
                f"{default_confidence}"
            )

        # Validate 'log' and 'is_fidelity' types to prevent configuration errors
        # from the YAML input
        for param, value in {"log": log, "is_fidelity": is_fidelity}.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected '{param}' to be a boolean, but got type: "
                    f"{type(value).__name__}"
                )

        self.lower: T = lower
        self.upper: T = upper
        self.log: bool = log
        self.domain: Domain[T] = domain
        self.log_bounds: tuple[float, float] | None = None
        self.log_default: float | None = None
        if self.log:
            self.log_bounds = (float(np.log(lower)), float(np.log(upper)))
            self.log_default = (
                float(np.log(self.default)) if self.default is not None else None
            )

        self.default_confidence_choice: Literal["low", "medium", "high"] = (
            default_confidence
        )

        self.default_confidence_score: float = self.DEFAULT_CONFIDENCE_SCORES[
            default_confidence
        ]
        self.has_prior: bool = self.default is not None

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return (
            self.lower == other.lower
            and self.upper == other.upper
            and self.log == other.log
            and self.is_fidelity == other.is_fidelity
            and self.value == other.value
            and self.default == other.default
            and self.default_confidence_score == other.default_confidence_score
        )

    def _get_truncnorm_prior_and_std(self) -> tuple[TruncNorm, float]:
        if self.log:
            assert self.log_bounds is not None
            low, high = self.log_bounds
            default = self.log_default
        else:
            low, high = self.lower, self.upper
            default = self.default

        assert default is not None
        return _get_truncnorm_prior_and_std(
            low=low,
            high=high,
            default=default,
            confidence_score=self.default_confidence_score,
        )


class NumericalParameter(Numerical):
    """Deprecated: Use `Numerical` instead of `NumericalParameter`.

    This class remains for backward compatibility and will raise a deprecation
    warning if used.
    """

    def __init__(
        self,
        lower: T,
        upper: T,
        *,
        log: bool = False,
        default: T | None,
        is_fidelity: bool,
        domain: Domain[T],
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        """Initialize a deprecated `NumericalParameter`.

        Args:
            lower: The lower bound of the numerical hyperparameter.
            upper: The upper bound of the numerical hyperparameter.
            log: Whether the hyperparameter is in log space.
            default: The default value of the hyperparameter.
            is_fidelity: Whether the hyperparameter is a fidelity parameter.
            domain: The domain of the hyperparameter.
            default_confidence: The default confidence choice.

        Raises:
            DeprecationWarning: A warning indicating that `neps.NumericalParameter` is
            deprecated and `neps.Numerical` should be used instead.
        """
        import warnings

        warnings.warn(
            (
                "Usage of 'neps.NumericalParameter' is deprecated and will be removed in"
                " future releases. Please use 'neps.Numerical' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            lower=lower,
            upper=upper,
            log=log,
            default=default,
            is_fidelity=is_fidelity,
            domain=domain,
            default_confidence=default_confidence,
        )
