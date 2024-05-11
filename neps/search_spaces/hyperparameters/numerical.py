"""The [`NumericalParameter`][neps.search_spaces.NumericalParameter] is
a [`Parameter`][neps.search_spaces.Parameter] that represents a numerical
range.

The two primary numerical hyperparameters are:

* [`FloatParameter`][neps.search_spaces.FloatParameter] for continuous
    float values.
* [`IntegerParameter`][neps.search_spaces.IntegerParameter] for discrete
    integer values.

The [`NumericalParameter`][neps.search_spaces.NumericalParameter] is a
base class for both of these hyperparameters, and includes methods from
both [`ParameterWithPrior`][neps.search_spaces.ParameterWithPrior],
allowing you to set a confidence along with a
[`.default`][neps.search_spaces.Parameter.default] that can be used
with certain algorithms, as well as
[`MutatableParameter`][neps.search_spaces.MutatableParameter],
which allows for [`mutate()`][neps.search_spaces.NumericalParameter.mutate]
and [`crossover()`][neps.search_spaces.NumericalParameter.crossover] operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Mapping, TypeVar
from typing_extensions import Self, override

import numpy as np
import scipy

from neps.search_spaces.parameter import MutatableParameter, ParameterWithPrior

if TYPE_CHECKING:
    from neps.search_spaces.hyperparameters.float import FloatParameter
    from neps.search_spaces.hyperparameters.integer import IntegerParameter
    from neps.types import TruncNorm

T = TypeVar("T", int, float)


class NumericalParameter(ParameterWithPrior[T, T], MutatableParameter):
    """A numerical hyperparameter is bounded by a lower and upper value.

    Attributes:
        lower: The lower bound of the numerical hyperparameter.
        upper: The upper bound of the numerical hyperparameter.
        log: Whether the hyperparameter is in log space.
        log_value: The log value of the hyperparameter, if `log=True`.
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
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        """Initialize the numerical hyperparameter.

        Args:
            lower: The lower bound of the numerical hyperparameter.
            upper: The upper bound of the numerical hyperparameter.
            log: Whether the hyperparameter is in log space.
            default: The default value of the hyperparameter.
            is_fidelity: Whether the hyperparameter is a fidelity parameter.
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
        self.log_value: float | None = None
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

    # TODO(eddiebergman): Right now this is identical for both float and integer
    # however the integer version is buggy as it returns duplicates due to rounding
    # Likely need to move this down to subclasses when addressed.
    @override
    def _get_neighbours(self, num_neighbours: int, *, std: float = 0.2) -> list[Self]:
        neighbours: list[Self] = []

        assert self.value is not None
        vectorized_val = self.value_to_normalized(self.value)

        # TODO(eddiebergman): This whole thing can be vectorized, not sure
        # if we ever have enough num_neighbours to make it worth it
        while len(neighbours) < num_neighbours:
            n_val = np.random.normal(vectorized_val, std)
            if n_val < 0 or n_val > 1:
                continue

            sampled_value = self.normalized_to_value(n_val)

            neighbour = self.clone()
            neighbour.set_value(sampled_value)
            neighbours.append(neighbour)

        return neighbours

    @override
    def compute_prior(self, *, log: bool = False) -> float:
        default = self.log_default if self.log else self.default

        assert self.value is not None
        assert default is not None

        value = np.log(self.value) if self.log else self.value
        value -= default
        dist, std = self._get_truncnorm_prior_and_std()
        value /= std
        prior = np.log(dist.pdf(value) + 1e-12) if log else dist.pdf(value)
        return float(prior)

    @override
    def mutate(
        self,
        parent: Self | None = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "local_search",
        **kwargs: Any,
    ) -> Self:
        if self.is_fidelity:
            raise ValueError("Trying to mutate fidelity param!")

        if parent is None:
            parent = self

        if mutation_strategy == "simple":
            child = self.clone()
            child.sample()
        elif mutation_strategy == "local_search" and "std" in kwargs:
            child = self._get_neighbours(std=kwargs["std"], num_neighbours=1)[0]
        elif mutation_strategy == "local_search":
            child = self._get_neighbours(num_neighbours=1)[0]
        else:
            raise NotImplementedError

        if parent.value == child.value:
            raise ValueError("Parent is the same as child!")

        return child

    @override
    def crossover(self, parent1: Self, parent2: Self | None = None) -> tuple[Self, Self]:
        if self.is_fidelity:
            raise ValueError("Trying to crossover fidelity param!")

        if parent2 is None:
            parent2 = self

        assert parent1.value is not None
        assert parent2.value is not None

        crossover_value = (parent1.value + parent2.value) / 2

        proxy_self = self.clone()
        proxy_self.set_value(crossover_value)  # type: ignore

        tt = tuple(proxy_self._get_neighbours(std=0.1, num_neighbours=2))
        assert len(tt) == 2
        return tt

    def _get_truncnorm_prior_and_std(self) -> tuple[TruncNorm, float]:
        if self.log:
            assert self.log_bounds is not None
            low, high = self.log_bounds
            default = self.log_default
        else:
            low, high = self.lower, self.upper
            default = self.default

        assert default is not None

        std = (high - low) * self.default_confidence_score
        a, b = (low - default) / std, (high - default) / std
        return scipy.stats.truncnorm(a, b), float(std)

    def to_integer(self) -> IntegerParameter:
        """Convert the numerical hyperparameter to an integer hyperparameter."""
        from neps.search_spaces.hyperparameters.integer import IntegerParameter

        as_int = lambda x: int(np.rint(x))

        int_hp = IntegerParameter(
            lower=as_int(self.lower),
            upper=as_int(self.upper),
            is_fidelity=self.is_fidelity,
            default=as_int(self.default) if self.default is not None else None,
            default_confidence=self.default_confidence_choice,  # type: ignore
        )
        int_hp.set_value(as_int(self.value) if self.value is not None else None)
        return int_hp

    def to_float(self) -> FloatParameter:
        """Convert the numerical hyperparameter to a float hyperparameter."""
        from neps.search_spaces.hyperparameters.integer import FloatParameter

        float_hp = FloatParameter(
            lower=float(self.lower),
            upper=float(self.upper),
            is_fidelity=self.is_fidelity,
            default=float(self.default) if self.default is not None else None,
            default_confidence=self.default_confidence_choice,  # type: ignore
        )
        float_hp.set_value(float(self.value) if self.value is not None else None)
        return float_hp

    def grid(self, *, size: int, include_endpoint: bool = True) -> list[T]:
        """Generate a grid of values for the numerical hyperparameter.

        !!! note "Duplicates"

            The grid may contain duplicates if the hyperparameter is an integer,
            for example if the lower bound is `0` and the upper bound is `10`, but
            `size=20`.

        Args:
            size: The number of values to generate.
            include_endpoint: Whether to include the upper bound in the grid.

        Returns:
            A list of values for the numerical hyperparameter.
        """
        return [
            self.normalized_to_value(x)
            for x in np.linspace(0, 1, num=size, endpoint=include_endpoint)
        ]

    @override
    @classmethod
    def serialize_value(cls, value: T) -> T:
        return value

    @override
    @classmethod
    def deserialize_value(cls, value: T) -> T:
        return value
