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
        self.has_prior = self.default is not None

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

    def _get_low_high_default(self):
        if self.log:
            return self._lower, self._upper, self._default
        else:
            return self.lower, self.upper, self.default

    def _get_truncnorm_prior_and_std(self):
        low, high, default = self._get_low_high_default()
        std = (high - low) * self.default_confidence_score
        a, b = (low - default) / std, (high - default) / std
        return scipy.stats.truncnorm(a, b), std

    def compute_prior(self, log: bool = False):
        _, _, default = self._get_low_high_default()
        value = np.log(self.value) if self.log else self.value
        value -= default
        dist, std = self._get_truncnorm_prior_and_std()
        value /= std
        return np.log(dist.pdf(value) + 1e-12) if log else dist.pdf(value)

    def sample(self, user_priors: bool = False):
        low, high, default = self._get_low_high_default()

        if user_priors and self.has_prior:
            dist, std = self._get_truncnorm_prior_and_std()
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
        if self.is_fidelity:
            raise ValueError("Trying to mutate fidelity param!")
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

        # child.value = min(1.0, max(0.0, child.value))
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

        if self.value is not None:
            self.value = (self.value - self.lower) / (self.upper - self.lower)

    def _inv_transform(self):
        if self.value != self.value:
            raise ValueError("Float parameter value is NaN!")

        if self.value is not None:
            self.value = self.value * (self.upper - self.lower) + self.lower
