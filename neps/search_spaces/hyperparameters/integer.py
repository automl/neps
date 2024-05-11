"""Float hyperparameter for search spaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Mapping
from typing_extensions import Self, override

import numpy as np

from neps.search_spaces.hyperparameters.float import FloatParameter
from neps.search_spaces.hyperparameters.numerical import NumericalParameter

if TYPE_CHECKING:
    from neps.utils.types import Number


class IntegerParameter(NumericalParameter[int]):
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
        lower = int(np.rint(lower))
        upper = int(np.rint(upper))
        _size = upper - lower + 1
        if _size <= 1:
            raise ValueError(
                f"IntegerParameter: expected at least 2 possible values in the range,"
                f" got upper={upper}, lower={lower}."
            )

        super().__init__(
            lower=int(np.rint(lower)),
            upper=int(np.rint(upper)),
            log=log,
            is_fidelity=is_fidelity,
            default=int(np.rint(default)) if default is not None else None,
            default_confidence=default_confidence,
        )

        # We subtract/add 0.499999 from lower/upper bounds respectively, such that
        # sampling in the float space gives equal probability for all integer values,
        # i.e. [x - 0.499999, x + 0.499999]
        self.float_hp = FloatParameter(
            lower=self.lower - 0.499999,
            upper=self.upper + 0.499999,
            log=self.log,
            is_fidelity=is_fidelity,
            default=default,
            default_confidence=default_confidence,
        )

    def __repr__(self) -> str:
        return f"<Integer, range: [{self.lower}, {self.upper}], value: {self.value}>"

    @override
    def clone(self) -> Self:
        clone = self.__class__(
            lower=self.lower,
            upper=self.upper,
            log=self.log,
            is_fidelity=self.is_fidelity,
            default=self.default,
            default_confidence=self.default_confidence_choice,
        )
        if self.value is not None:
            clone.set_value(self.value)

        return clone

    @override
    def load_from(self, value: Number) -> None:
        self._value = int(np.rint(value))

    @override
    def set_default(self, default: int | None) -> None:
        if default is None:
            self.default = None
            self.has_prior = False
            self.float_hp.set_default(None)
        else:
            _default = int(round(default))
            self.default = _default
            self.has_prior = True
            self.float_hp.set_default(_default)

    @override
    def set_value(self, value: int | None) -> None:
        if value is None:
            self._value = None
            self.normalized_value = None
            self.log_value = None
            self.float_hp.set_value(None)
            return

        if not self.lower <= value <= self.upper:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"{cls_name} parameter: default bounds error. Expected lower <= default"
                f" <= upper, but got lower={self.lower}, value={value},"
                f" upper={self.upper}"
            )

        value = int(np.rint(value))

        self.float_hp.set_value(value)
        self._value = value
        self.normalized_value = self.value_to_normalized(value)
        if self.log:
            self.log_value = np.log(value)

    @override
    def sample_value(self, *, user_priors: bool = False) -> int:
        val = self.float_hp.sample_value(user_priors=user_priors)
        return int(np.rint(val))

    @override
    def value_to_normalized(self, value: int) -> float:
        return self.float_hp.value_to_normalized(float(np.rint(value)))

    @override
    def normalized_to_value(self, normalized_value: float) -> int:
        return int(np.rint(self.float_hp.normalized_to_value(normalized_value)))

    @override
    def set_default_confidence_score(self, default_confidence: str) -> None:
        self.float_hp.set_default_confidence_score(default_confidence)
        super().set_default_confidence_score(default_confidence)

    @override
    def _get_non_unique_neighbors(
        self,
        num_neighbours: int,
        *,
        std: float = 0.2,
    ) -> list[Self]:
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
            if sampled_value == self.value:
                continue

            neighbour = self.clone()
            neighbour.set_value(sampled_value)
            neighbours.append(neighbour)

        return neighbours
