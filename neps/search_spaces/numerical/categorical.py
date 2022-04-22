from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .numerical import NumericalParameter


class CategoricalParameter(NumericalParameter):
    def __init__(
        self,
        choices: Iterable[float | int | str],
        default: None | float | int | str = None,
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        super().__init__()

        self.default = default
        self.default_confidence_score = dict(low=1.1, medium=1.75, high=2.5)[
            default_confidence
        ]
        self.has_prior = self.default is not None

        self.choices = list(choices)
        self.num_choices = len(self.choices)
        self.probabilities: list[npt.NDArray] = list(
            np.ones(self.num_choices) * (1.0 / self.num_choices)
        )
        self.value: None | float | int | str = None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.choices == other.choices
            and self.value == other.value
        )

    def __hash__(self):
        return hash((self.name, tuple(self.choices), self.value))

    def __repr__(self):
        return f"Categorical, choices: {self.choices}, value: {self.value}"

    def __copy__(self):
        return self.__class__(choices=self.choices)

    def _compute_user_prior_probabilities(self):
        # The default value should have "default_confidence_score" more probability than
        # all the other values.
        base_probability = 1 / (self.num_choices - 1 + self.default_confidence_score)
        probabilities = [base_probability] * self.num_choices
        default_index = self.choices.index(self.default)  # type: ignore[arg-type]
        probabilities[default_index] *= self.default_confidence_score
        return probabilities

    def compute_prior(self, log: bool = False):
        probabilities = self._compute_user_prior_probabilities()
        own_value_index = self.choices.index(self.value)  # type: ignore[arg-type]
        return (
            np.log(probabilities[own_value_index] + 1e-12)
            if log
            else probabilities[own_value_index]
        )

    def sample(self, user_priors: bool = False):
        if user_priors and self.default is not None:
            probabilities = self._compute_user_prior_probabilities()
        else:
            probabilities = self.probabilities

        idx = np.random.choice(a=self.num_choices, replace=True, p=probabilities)
        self.value = self.choices[int(idx)]

    def mutate(
        self,
        parent=None,
        mutation_rate: float = 1.0,  # pylint: disable=unused-argument
        mutation_strategy: str = "local_search",
    ):

        if parent is None:
            parent = self

        if mutation_strategy == "simple":
            child = self.__copy__()
            child.sample()
        elif mutation_strategy == "local_search":
            child = self._get_neighbours(num_neighbours=1)[0]
        else:
            raise NotImplementedError

        if parent.value == child.value:
            raise ValueError("Parent is the same as child!")

        return child

    def crossover(self, parent1, parent2=None):
        pass

    def _get_neighbours(self, num_neighbours: int = 1):
        neighbours: list[CategoricalParameter] = []

        idx = 0
        choices = self.choices.copy()
        random.shuffle(choices)

        while len(neighbours) < num_neighbours:
            if num_neighbours > self.num_choices - 1:
                choice = self.choices[np.random.randint(0, self.num_choices)]
            else:
                choice = choices[idx]
                idx += 1
            if choice == self.value and len(self.choices) > 1:
                continue
            neighbour = self.__copy__()
            neighbour.value = choice
            neighbours.append(neighbour)

        return neighbours

    def _transform(self):
        self.value = self.choices.index(self.value) / self.num_choices

    def _inv_transform(self):
        self.value = self.choices[int(self.value * self.num_choices)]
