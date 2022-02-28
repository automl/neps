from __future__ import annotations

import math

import numpy as np
import scipy.stats
from typing_extensions import Literal

from .numerical import NumericalParameter


class FloatParameter(NumericalParameter):
    def __init__(
        self,
        lower: float | int,
        upper: float | int,
        log: bool = False,
        is_fidelity: bool = False,
        default: None | float | int = None,
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        super().__init__()

        self.default = default
        self.default_confidence_score = dict(low=0.5, medium=0.25, high=0.125)[
            default_confidence
        ]

        self.is_fidelity = is_fidelity

        self.lower = float(lower)
        self.upper = float(upper)

        if self.lower >= self.upper:
            raise ValueError("Float parameter: bounds error (lower >= upper).")

        self.log = log

        if self.log:
            if self.lower <= 0:
                raise ValueError("Float parameter: bounds error (log scale).")
            self._lower = np.log(self.lower)
            self._upper = np.log(self.upper)
            if self.default is not None:
                self._default = np.log(self.default)
            else:
                self._default = None

        self.value: None | float = None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.lower == other.lower
            and self.upper == other.upper
            and self.log == other.log
            and self.value == other.value
        )

    def __hash__(self):
        return hash((self.lower, self.upper, self.log, self.value))

    def __repr__(self):
        return f"Float, range: [{self.lower}, {self.upper}], value: {self.value:.07f}"

    def __copy__(self):
        return self.__class__(lower=self.lower, upper=self.upper, log=self.log)

    def sample(self, use_user_priors: bool = False):  # pylint: disable=unused-argument
        if self.log:
            low, high, default = self._lower, self._upper, self._default
        else:
            low, high, default = self.lower, self.upper, self.default

        if use_user_priors and default is not None:
            std = (high - low) * self.default_confidence_score
            a, b = (low - default) / std, (high - default) / std
            dist = scipy.stats.truncnorm(a, b)  # dist.pdf(x) for pibo acq
            value = dist.rvs() * std + default
        else:
            value = np.random.uniform(low=low, high=high)

        if self.log:
            value = math.exp(value)

        self.value = min(self.upper, max(self.lower, value))

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
            child = self._get_neighbours(num_neighbours=1)[
                0
            ]  # pylint: disable=protected-access
        else:
            raise NotImplementedError

        if parent.value == child.value:
            raise ValueError("Parent is the same as child!")

        return child

    def crossover(self, parent1, parent2=None):
        raise NotImplementedError

    def _get_neighbours(self, std: float = 0.2, num_neighbours: int = 1):
        neighbours: list[FloatParameter] = []
        self._transform()  # pylint: disable=protected-access

        while len(neighbours) < num_neighbours:
            n_val = np.random.normal(self.value, std)
            if n_val < 0 or n_val > 1:
                continue
            neighbour = self.__copy__()
            neighbour.value = n_val
            neighbour._inv_transform()  # pylint: disable=protected-access
            neighbours.append(neighbour)

        self._inv_transform()  # pylint: disable=protected-access
        return neighbours

    def _transform(self):
        if self.value != self.value:
            raise ValueError("Float parameter value is NaN!")

        self.value = (self.value - self.lower) / (self.upper - self.lower)

    def _inv_transform(self):
        if self.value != self.value:
            raise ValueError("Float parameter value is NaN!")

        self.value = self.value * (self.upper - self.lower) + self.lower

    def create_from_id(self, identifier):
        self.value = identifier
