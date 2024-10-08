"""Categorical hyperparameter for search spaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias
from typing_extensions import Self, override

import numpy as np
import numpy.typing as npt
from more_itertools import all_unique

from neps.search_spaces.domain import Domain
from neps.search_spaces.parameter import MutatableParameter, ParameterWithPrior

if TYPE_CHECKING:
    from neps.utils.types import f64

CategoricalTypes: TypeAlias = float | int | str


class CategoricalParameter(
    ParameterWithPrior[CategoricalTypes, CategoricalTypes],
    MutatableParameter,
):
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

    Please see the [`Parameter`][neps.search_spaces.parameter],
    [`ParameterWithPrior`][neps.search_spaces.parameter.ParameterWithPrior],
    [`MutatableParameter`][neps.search_spaces.parameter.MutatableParameter] classes
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
        """Create a new `CategoricalParameter`.

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
    def compute_prior(self, log: bool = False) -> float:
        assert self._value_index is not None

        probabilities = self._compute_user_prior_probabilities()
        return float(
            np.log(probabilities[self._value_index] + 1e-12)
            if log
            else probabilities[self._value_index]
        )

    @override
    def sample_value(self, *, user_priors: bool = False) -> Any:
        indices = np.arange(len(self.choices))
        if user_priors and self.default is not None:
            probabilities = self._compute_user_prior_probabilities()
            return self.choices[np.random.choice(indices, p=probabilities)]

        return self.choices[np.random.choice(indices)]

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
            child = parent.sample()
        elif mutation_strategy == "local_search":
            child = self._get_non_unique_neighbors(num_neighbours=1)[0]
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

        child1 = parent1.clone()
        child1.set_value(parent2.value)

        child2 = parent2.clone()
        child2.set_value(parent1.value)

        return child1, child2

    @override
    def _get_non_unique_neighbors(
        self,
        num_neighbours: int,
        *,
        std: float = 0.2,
    ) -> list[Self]:
        assert self._value_index is not None

        indices = np.arange(len(self.choices))
        bot = indices[: self._value_index]
        top = indices[self._value_index + 1 :]
        available_neighbours = np.concatenate([bot, top])

        selected_indices = np.random.choice(
            available_neighbours,
            size=num_neighbours,
            replace=True,
        )

        new_neighbours: list[Self] = []
        for value_index in selected_indices:
            new_param = self.clone()
            new_param.set_value(self.choices[value_index])
            new_neighbours.append(new_param)

        return new_neighbours

    @override
    def value_to_normalized(self, value: Any) -> float:
        return float(self.choices.index(value))

    @override
    def normalized_to_value(self, normalized_value: float) -> Any:
        return self.choices[int(np.rint(normalized_value))]

    @override
    def set_default(self, default: Any | None) -> None:
        if default is None:
            self.default = None
            self._default_index = None
            self.has_prior = False
            return

        if default not in self.choices:
            raise ValueError(
                f"Default value {default} is not in the provided choices {self.choices}"
            )

        self.default = default
        self._default_index = self.choices.index(default)
        self.has_prior = True

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

    @override
    @classmethod
    def serialize_value(cls, value: CategoricalTypes) -> CategoricalTypes:
        return value

    @override
    @classmethod
    def deserialize_value(cls, value: CategoricalTypes) -> CategoricalTypes:
        return value
