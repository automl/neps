from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import Literal

from ..parameter import HpTensorShape, Parameter

CATEGORICAL_CONFIDENCE_SCORES = {
    "low": 2,
    "medium": 4,
    "high": 6,
}


class CategoricalParameter(Parameter):
    def __init__(
        self,
        choices: Iterable[float | int | str],
        is_fidelity: bool = False,
        default: None | float | int | str = None,
        default_confidence: Literal["low", "medium", "high"] = "low",
        **kwargs,
    ):
        self.choices: list[float | int | str] = list(choices)

        super().__init__(**kwargs)
        if len(self.choices) == 0:
            raise ValueError("Can't create a categorical parameter without choices")

        self.default = default
        self.lower = default
        self.upper = default
        self.default_confidence_score = CATEGORICAL_CONFIDENCE_SCORES[default_confidence]
        self.has_prior = self.default is not None

        self.is_fidelity = is_fidelity

        self.num_choices = len(self.choices)
        self.probabilities: list[npt.NDArray] = list(
            np.ones(self.num_choices) * (1.0 / self.num_choices)
        )
        self.value: None | float | int | str = None

    @property
    def id(self):
        return self.value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.choices == other.choices and self.value == other.value

    def __repr__(self):
        return f"<Categorical, choices: {self.choices}, value: {self.value}>"

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

        idx = np.random.choice(a=self.num_choices, p=probabilities)
        self.value = self.choices[int(idx)]

    def prior_probability(self):
        if self.default is not None:
            i_value = self.choices.index(self.value)
            return self._compute_user_prior_probabilities()[i_value]
        return 1

    def mutate(
        self,
        parent=None,
        mutation_rate: float = 1.0,  # pylint: disable=unused-argument
        mutation_strategy: str = "local_search",
    ):
        if self.is_fidelity:
            raise ValueError("Trying to mutate fidelity param!")

        if parent is None:
            parent = self

        if mutation_strategy == "simple":
            child = self.copy()
            child.sample()
        elif mutation_strategy == "local_search":
            child = self._get_neighbours(num_neighbours=1)[0]
        else:
            raise NotImplementedError

        if parent.value == child.value:
            raise ValueError("Parent is the same as child!")

        return child

    def crossover(self, parent1, parent2=None):
        if self.is_fidelity:
            raise ValueError("Trying to crossover fidelity param!")
        if parent2 is None:
            parent2 = self

        child1 = parent1.copy()
        child2 = parent2.copy()

        child1.value = parent2.value
        child2.value = parent1.value

        children = [child1, child2]

        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        # expected len(children) == num_neighbours
        return children

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
            neighbour = self.copy()
            neighbour.value = choice
            neighbours.append(neighbour)

        return neighbours

    def normalized(self):
        hp = CategoricalParameter(
            choices=list(range(len(self.choices))),
            is_fidelity=self.is_fidelity,
        )
        if self.value is not None:
            hp.value = self.choices.index(self.value)
        return hp

    def serialize(self):
        return self.value

    def load_from(self, value):
        self.value = value

    @staticmethod
    def get_tensor_shape(hp_instances):
        return HpTensorShape(1, hp_instances)

    def get_tensor_value(self, tensor_shape):  # pylint: disable=unused-argument
        return torch.tensor(self.normalized().value, dtype=torch.get_default_dtype())
