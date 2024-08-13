from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from neps.search_spaces.domain import Domain
from neps.utils.types import f64, i64

if TYPE_CHECKING:
    from neps.search_spaces.search_space import Confidence

T = TypeVar("T")
V = TypeVar("V", i64, f64)
N = TypeVar("N", i64, f64)


@dataclass
class NumericalParameter(ABC, Generic[V]):
    domain: Domain[V] = field(init=False, repr=False)


@dataclass
class FloatParameter(NumericalParameter[f64]):
    """A float parameter.

    ```python
    import neps

    l2_norm = neps.FloatParameter(0, 1)
    learning_rate = neps.FloatParameter(1e-4, 1e-1, log=True)
    ```

    Args:
        lower: lower bound for the hyperparameter.
        upper: upper bound for the hyperparameter.
        log: whether the hyperparameter is on a log scale.
        is_fidelity: whether the hyperparameter is fidelity.
        q: quantization factor for the hyperparameter. Will round values
            to the nearest multiple of `q`. Mutually exclusive of both `bins` and `log`.
        bins: number of bins to quantize the hyperparameter into. Will round
            values to the nearest bin. Mutually exclusive. Can be used with `log`
        default: default value for the hyperparameter.
        default_confidence: confidence score for the default value, used when
            condsidering prior based optimization..
    """

    lower: float
    upper: float
    log: bool = False
    q: float | None = None
    bins: int | None = None

    # There should likely exist in the SearchSpace definition, not the parameter
    is_fidelity: bool = False
    default: float | None = None
    default_confidence: Confidence = "low"

    def __post_init__(self):
        # If quantized, round the bounds to the nearest multiple of q
        if self.q:
            if self.log:
                raise ValueError(
                    f"Cannot specify both q and log for parameter {self}!"
                    " You can however specify `bins` for similar effect."
                )
            if self.bins:
                raise ValueError("Cannot specify both q and bins!")

            lower = np.rint(self.lower / self.q) * self.q
            upper = np.rint(self.upper / self.q) * self.q
            bins = int(np.rint((upper - lower) / self.q)) + 1
        elif self.bins:
            lower = self.lower
            upper = self.upper
            bins = self.bins
        else:
            lower = self.lower
            upper = self.upper
            bins = None

        self.domain = Domain.float(lower, upper, log=self.log, bins=bins)
        self.lower = float(lower)
        self.upper = float(upper)
        if self.default is not None:
            self.default = float(self.default)
            # TODO(eddiebergman): Maybe we should just ignored and use the
            # closest value in the domain?
            closest_default = self.domain.closest(f64(self.default))
            if not np.isclose(self.default, closest_default):
                raise ValueError(
                    f"Default value {self.default} is not in the domain of {self},"
                    f" the closest value in the domain is {closest_default}. Common"
                    " reasons could be due to quantization or binning."
                )

        if self.default_confidence not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid confidence: {self.default_confidence}")


@dataclass
class IntegerParameter(NumericalParameter[i64]):
    """An integer parameter.

    ```python
    import neps

    batch_size = neps.IntegerParameter(32, 128)
    num_layers = neps.IntegerParameter(1, 1000, log=True)
    ```
    Args:
        lower: lower bound for the hyperparameter.
        upper: upper bound for the hyperparameter.
        log: whether the hyperparameter is on a log scale.
        q: quantization factor for the hyperparameter. Will round values
            to the nearest multiple of `q`. Mutually exclusive of both `bins` and `log`.
        bins: number of bins to quantize the hyperparameter into. Will round
            values to the nearest bin. Mutually exclusive. Can be used with `log`
        is_fidelity: whether the hyperparameter is fidelity.
        default: default value for the hyperparameter.
        default_confidence: confidence score for the default value, used when
            condsider prior based optimization.
    """

    lower: int
    upper: int
    log: bool = False
    q: int | None = None
    bins: int | None = None

    # There should likely exist in the SearchSpace definition, not the parameter
    is_fidelity: bool = False
    default: float | None = None
    default_confidence: Confidence = "low"

    def __post_init__(self):
        # If quantized, round the bounds to the nearest multiple of q
        if self.q:
            if self.log:
                raise ValueError(
                    f"Cannot specify both q and log for parameter {self}!"
                    " You can however specify `bins` for similar effect."
                )
            if self.bins:
                raise ValueError("Cannot specify both q and bins!")

            lower = np.rint(self.lower / self.q) * self.q
            upper = np.rint(self.upper / self.q) * self.q
            bins = int(np.rint((upper - lower) / self.q)) + 1
        elif self.bins:
            lower = self.lower
            upper = self.upper
            bins = self.bins
        else:
            lower = self.lower
            upper = self.upper
            bins = None

        self.domain = Domain.int(lower, upper, log=self.log, bins=bins)
        self.lower = round(lower)
        self.upper = round(upper)
        if self.default is not None:
            self.default = round(self.default)
            # TODO(eddiebergman): Maybe we should just ignored and use the
            # closest value in the domain?
            closest_default = self.domain.closest(i64(self.default))
            if not np.isclose(self.default, closest_default):
                raise ValueError(
                    f"Default value {self.default} is not in the domain of {self},"
                    f" the closest value in the domain is {closest_default}. Common"
                    " reasons could be due to quantization or binning."
                )

        if self.default_confidence not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid confidence: {self.default_confidence}")
