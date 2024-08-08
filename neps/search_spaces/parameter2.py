from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    Mapping,
    TypeAlias,
    TypedDict,
    TypeVar,
)
from typing_extensions import override

import numpy as np

from neps.env import ALLOW_LARGE_CATEGORIES
from neps.search_spaces.distributions import (
    UNIT_UNIFORM,
    Distribution,
    TruncNormDistribution,
    UniformIntDistribution,
    WeightedIntsDistribution,
)
from neps.search_spaces.domain import Domain
from neps.utils.types import Array, f64, i64

if TYPE_CHECKING:
    from neps.utils.types import Number

T = TypeVar("T")
V = TypeVar("V", i64, f64)
S = TypeVar("S", i64, f64)

# Should probably replace this with an enum
Confidence: TypeAlias = Literal["low", "medium", "high"]


class ConfidenceMapping(TypedDict):
    low: float
    medium: float
    high: float


class Parameter(ABC, Generic[T, V]):
    domain: Domain[V]
    is_fidelity: bool
    default: T | None
    default_confidence: Confidence
    default_confidence_score: float

    DEFAULT_CONFIDENCE_SCORES: ClassVar[ConfidenceMapping]

    @abstractmethod
    def to_vector(self, lst: list[T]) -> Array[V]: ...

    @abstractmethod
    def to_values(self, arr: Array[V]) -> list[T]: ...

    @abstractmethod
    def uniform_distribution(self) -> Distribution: ...

    @abstractmethod
    def prior_distribution(self, value: T | V, confidence: float) -> Distribution: ...


class FloatParameter(Parameter[float, f64]):
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
    """

    domain: Domain[f64]
    default: float | None
    default_confidence: Confidence
    is_fidelity: bool

    DEFAULT_CONFIDENCE_SCORES: ClassVar[ConfidenceMapping] = {
        "low": 0.5,
        "medium": 0.75,
        "high": 0.875,
    }

    def __init__(
        self,
        lower: Number,
        upper: Number,
        *,
        log: bool = False,
        is_fidelity: bool = False,
        default: Number | None = None,
        default_confidence: Confidence = "low",
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
        self.is_fidelity = is_fidelity
        self.default = float(default) if default is not None else None
        self.default_confidence = default_confidence
        self.domain = Domain.float(lower, upper, log=log)

    @override
    def to_values(self, arr: Array[f64]) -> list[float]:
        return arr.tolist()

    @override
    def to_vector(self, lst: list[float]) -> Array[f64]:
        return np.array(lst, dtype=f64)

    @override
    def uniform_distribution(self) -> Distribution:
        return UNIT_UNIFORM

    @override
    def prior_distribution(self, value: Number, confidence: float) -> Distribution:
        assert 0 <= confidence <= 1
        return TruncNormDistribution.new(
            mean=f64(value),
            std=(1 - confidence),
            lower=self.domain.lower,
            upper=self.domain.upper,
        )


class IntegerParameter(Parameter[int, i64]):
    """An integer value for a parameter.

    This kind of [`Parameter`][neps.search_spaces.parameter] is used
    to represent hyperparameters with continuous integer values, optionally specifying
    f it exists on a log scale.
    For example, `batch_size` could be a value in `(32, 128)`, while the `num_layers`
    hyperparameter in a neural network search space can be a `IntegerParameter`
    with a range of `(1, 1000)` but on a log scale.

    ```python
    import neps

    batch_size = neps.IntegerParameter(32, 128)
    num_layers = neps.IntegerParameter(1, 1000, log=True)
    ```
    """

    domain: Domain[i64]
    default: int | None
    default_confidence: Confidence
    is_fidelity: bool

    DEFAULT_CONFIDENCE_SCORES: ClassVar[ConfidenceMapping] = {
        "low": 0.5,
        "medium": 0.75,
        "high": 0.875,
    }

    def __init__(
        self,
        lower: Number,
        upper: Number,
        *,
        log: bool = False,
        is_fidelity: bool = False,
        default: Number | None = None,
        default_confidence: Confidence = "low",
    ):
        """Create a new `IntegerParameter`.

        Args:
            lower: lower bound for the hyperparameter.
            upper: upper bound for the hyperparameter.
            log: whether the hyperparameter is on a log scale.
            is_fidelity: whether the hyperparameter is fidelity.
            default: default value for the hyperparameter.
            default_confidence: confidence score for the default value, used when
                condsider prior based optimization.
        """
        self.is_fidelity = is_fidelity
        self.default = int(np.rint(default)) if default is not None else None
        self.default_confidence = default_confidence
        self.domain = Domain.int(lower, upper, log=log)

    @override
    def to_list(self, arr: Array[i64]) -> list[int]:
        return arr.tolist()

    @override
    def to_vector(self, lst: list[int]) -> Array[i64]:
        return np.array(lst, dtype=i64)

    @override
    def uniform_distribution(self) -> Distribution:
        return UNIT_UNIFORM

    @override
    def prior_distribution(self, value: Number, confidence: float) -> Distribution:
        assert 0 <= confidence <= 1
        return TruncNormDistribution.new(
            mean=f64(value),
            std=(1 - confidence),
            lower=self.domain.lower,
            upper=self.domain.upper,
        )


# TODO(eddiebergman): DEFAULT_CONFIDENCE has the problem that the influence of
# this is highly dependant on the amount of values.
# Take for example a categorical with 2 values or 100 values.
# With "high" confidence, we will select it frequently for the 2 values
# but with 100 values, this drops to near 0 as well.
# However we are very unlikely to be in a scenario where we have 100 categories
class CategoricalParameter(Parameter[Any, i64]):
    """A list of **unordered** choices for a parameter.

    This kind of [`Parameter`][neps.search_spaces.parameter] is used
    to represent hyperparameters that can take on a discrete set of unordered
    values. For example, the `optimizer` hyperparameter in a neural network
    search space can be a `CategoricalParameter` with choices like
    `#!python ["adam", "sgd", "rmsprop"]`.

    ```python
    import neps

    optimizer_choice = neps.CategoricalParameter(
        ["adam", "sgd", "rmsprop"],
        default="adam"
    )
    ```
    """

    CATEGORY_SIZE_TO_PREFER_HASH: ClassVar[int] = 10
    LARGE_CATEGORY_LIMIT: ClassVar[int] = 20

    DEFAULT_CONFIDENCE_SCORES: ClassVar[Mapping[str, Any]] = {
        "low": 0.5,
        "medium": 0.75,
        "high": 0.875,
    }

    domain: Domain[i64]
    default_confidence: Confidence
    default: Any | None
    is_fidelity: Literal[False]

    def __init__(
        self,
        choices: Iterable[float | int | str],
        *,
        default: float | int | str | None = None,
        default_confidence: Confidence = "low",
    ):
        """Create a new `CategoricalParameter`.

        Args:
            choices: choices for the hyperparameter.
            default: default value for the hyperparameter, must be in `choices=`
                if provided.
            default_confidence: confidence score for the default value, used when
                condsider prior based optimization.
        """
        choices = tuple(choices)
        if not ALLOW_LARGE_CATEGORIES and len(choices) > self.LARGE_CATEGORY_LIMIT:
            raise ValueError(
                f"NePS was not designed to handle more than {self.LARGE_CATEGORY_LIMIT}"
                " categories. Many operations will be slow or fail. Please let us know"
                " your use case! To remove this restriction, set the env var"
                " `NEPS_ALLOW_LARGE_CATEGORIES=1` or set"
                " `CategoricalParameter.LARGE_CATEGORY_LIMIT` to at"
                f" least {len(choices)} to acommodate this parameter."
            )
        self.domain = Domain.int(0, len(choices))
        self.default_confidence = default_confidence
        self.default = default
        self.is_fidelity = False
        self._choices = choices

        self._lookup: dict[Any, int] | None = None

        # Prefer hash lookup for large categories, otherwise an interated
        # index lookup should be faster
        if len(self._choices) > self.CATEGORY_SIZE_TO_PREFER_HASH:
            try:
                self._lookup = {v: i for i, v in enumerate(choices)}
            except TypeError:
                self._lookup = None
        else:
            self._lookup = None

    @override
    def to_values(self, arr: Array[i64]) -> list[Any]:
        return [self._choices[i] for i in arr.tolist()]

    @override
    def to_vector(self, lst: list[Any]) -> Array[i64]:
        if self._lookup is not None:
            return np.array([self._lookup[v] for v in lst], dtype=i64)
        return np.array([self._choices.index(v) for v in lst], dtype=i64)

    @override
    def uniform_distribution(self) -> Distribution:
        return UniformIntDistribution.new(0, len(self._choices) - 1)

    @override
    def prior_distribution(self, value: Any, confidence: float) -> Distribution:
        assert 0 <= confidence <= 1
        if self._lookup is not None:
            _index = self._lookup[value]
        else:
            _index = self._choices.index(value)

        remaining_weight = 1 - confidence
        weights = np.full(len(self._choices), remaining_weight)
        weights[_index] = confidence
        return WeightedIntsDistribution.new(weights)


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
