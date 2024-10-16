"""Categorical hyperparameter for search spaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias
from typing_extensions import Self, override

import numpy as np
import numpy.typing as npt
from more_itertools import all_unique

from neps.search_spaces.domain import Domain
from neps.search_spaces.parameter import ParameterWithPrior

if TYPE_CHECKING:
    from neps.utils.types import f64

CategoricalTypes: TypeAlias = float | int | str


class Categorical(ParameterWithPrior[CategoricalTypes, CategoricalTypes]):
    """A list of **unordered** choices for a parameter.

    This kind of [`Parameter`][neps.search_spaces.parameter] is used
    to represent hyperparameters that can take on a discrete set of unordered
    values. For example, the `optimizer` hyperparameter in a neural network
    search space can be a `Categorical` with choices like
    `#!python ["adam", "sgd", "rmsprop"]`.

    ```python
    import neps

    optimizer_choice = neps.Categorical(
        ["adam", "sgd", "rmsprop"],
        default="adam"
    )
    ```

    Please see the [`Parameter`][neps.search_spaces.parameter],
    [`ParameterWithPrior`][neps.search_spaces.parameter.ParameterWithPrior],
    for more details on the methods available for this class.
    """

    DEFAULT_CONFIDENCE_SCORES: ClassVar[Mapping[str, Any]] = {
        "low": 2,
        "medium": 4,
        "high": 6,
    }

    def __init__(
        self,
        choices: Iterable[float | int | str],
        *,
        default: float | int | str | None = None,
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        """Create a new `Categorical`.

        Args:
            choices: choices for the hyperparameter.
            default: default value for the hyperparameter, must be in `choices=`
                if provided.
            default_confidence: confidence score for the default value, used when
                condsider prior based optimization.
        """
        choices = list(choices)
        if len(choices) <= 1:
            raise ValueError("Categorical choices must have more than one value.")

        super().__init__(value=None, is_fidelity=False, default=default)

        for choice in choices:
            if not isinstance(choice, float | int | str):
                raise TypeError(
                    f'Choice "{choice}" is not of a valid type (float, int, str)'
                )

        if not all_unique(choices):
            raise ValueError(f"Choices must be unique but got duplicates.\n{choices}")

        if default is not None and default not in choices:
            raise ValueError(
                f"Default value {default} is not in the provided choices {choices}"
            )

        self.choices = list(choices)

        # NOTE(eddiebergman): If there's ever a very large categorical,
        # then it would be beneficial to have a lookup table for indices as
        # currently we do a list.index() operation which is O(n).
        # However for small sized categoricals this is likely faster than
        # a lookup table.
        # For now we can just cache the index of the value and default.
        self._value_index: int | None = None

        self.default_confidence_choice = default_confidence
        self.default_confidence_score = self.DEFAULT_CONFIDENCE_SCORES[default_confidence]
        self.has_prior = self.default is not None
        self._default_index: int | None = (
            self.choices.index(default) if default is not None else None
        )
        self.domain = Domain.indices(len(self.choices))

    @override
    def clone(self) -> Self:
        clone = self.__class__(
            choices=self.choices,
            default=self.default,
            default_confidence=self.default_confidence_choice,  # type: ignore
        )
        if self.value is not None:
            clone.set_value(self.value)

        return clone

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return (
            self.choices == other.choices
            and self.value == other.value
            and self.is_fidelity == other.is_fidelity
            and self.default == other.default
            and self.has_prior == other.has_prior
            and self.default_confidence_score == other.default_confidence_score
        )

    def __repr__(self) -> str:
        return f"<Categorical, choices: {self.choices}, value: {self.value}>"

    def _compute_user_prior_probabilities(self) -> npt.NDArray[f64]:
        # The default value should have "default_confidence_score" more probability
        # than all the other values.
        assert self._default_index is not None
        probabilities = np.ones(len(self.choices))
        probabilities[self._default_index] = self.default_confidence_score
        return probabilities / np.sum(probabilities)

    @override
    def sample_value(self, *, user_priors: bool = False) -> Any:
        indices = np.arange(len(self.choices))
        if user_priors and self.default is not None:
            probabilities = self._compute_user_prior_probabilities()
            return self.choices[np.random.choice(indices, p=probabilities)]

        return self.choices[np.random.choice(indices)]

    @override
    def value_to_normalized(self, value: Any) -> float:
        return float(self.choices.index(value))

    @override
    def normalized_to_value(self, normalized_value: float) -> Any:
        return self.choices[int(np.rint(normalized_value))]

    @override
    def set_value(self, value: Any | None) -> None:
        if value is None:
            self._value = None
            self._value_index = None
            self.normalized_value = None
            return

        self._value = value
        self._value_index = self.choices.index(value)
        self.normalized_value = float(self._value_index)


class CategoricalParameter(Categorical):
    """Deprecated: Use `Categorical` instead of `CategoricalParameter`.

    This class remains for backward compatibility and will raise a deprecation
    warning if used.
    """

    def __init__(
        self,
        choices: Iterable[float | int | str],
        *,
        default: float | int | str | None = None,
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        """Initialize a deprecated `CategoricalParameter`.

        Args:
            choices: choices for the hyperparameter.
            default: default value for the hyperparameter, must be in `choices=`
                if provided.
            default_confidence: confidence score for the default value, used when
                condsider prior based optimization.

        Raises:
            DeprecationWarning: A warning indicating that `neps.CategoricalParameter` is
            deprecated and `neps.Categorical` should be used instead.
        """
        import warnings

        warnings.warn(
            (
                "Usage of 'neps.CategoricalParameter' is deprecated and will be removed "
                "in future releases. Please use 'neps.Categorical' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            choices=choices,
            default=default,
            default_confidence=default_confidence,
        )
