from __future__ import annotations

import math

import numpy as np
import scipy.stats
from typing_extensions import Literal

from .numerical import NumericalParameter

FLOAT_CONFIDENCE_SCORES = {
    "low": 0.5,
    "medium": 0.25,
    "high": 0.125,
}


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
        super().__init__(is_fidelity=is_fidelity)

        self.default = default
        self.default_confidence_score = FLOAT_CONFIDENCE_SCORES[default_confidence]
        self.has_prior = self.default is not None

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

    def __repr__(self):
        float_repr = f"{self.value:.07f}" if self.value is not None else "None"
        return f"<Float, range: [{self.lower}, {self.upper}], value: {float_repr}>"

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

    def step_on_scale(self, scale, use_value=None):
        """Returns the corresponding step on a scale from 0 to scale-1"""
        assert scale >= 1
        low, high, _ = self._get_low_high_default()
        value = self.value if use_value is None else use_value
        value = math.log(value) if self.log else value
        float_fraction = (value - low) / high
        return int(round(float_fraction * (scale - 1)))

    def from_step(self, step, scale, in_place=True):
        """Sets the value corresponding to a step on a scale from 0 to scale-1"""
        assert scale >= 1 and step >= 0 and step < scale
        low, high, _ = self._get_low_high_default()
        fract = step / (scale - 1) if scale > 1 else 0
        value = low + (high - low) * fract
        value = math.exp(value) if self.log else value
        if in_place:
            self.value = value
        return value

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
        if self.is_fidelity:
            raise ValueError("Trying to crossover fidelity param!")
        if parent2 is None:
            parent2 = self

        proxy_self = self.copy()
        proxy_self.value = (parent1.value + parent2.value) / 2
        # pylint: disable=protected-access
        children = proxy_self._get_neighbours(std=0.1, num_neighbours=2)

        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        # expected len(children) == num_neighbours
        return children

    def _get_neighbours(self, std: float = 0.2, num_neighbours: int = 1):
        neighbours: list[FloatParameter] = []
        cur_value = self._normalize_value(self.value)

        while len(neighbours) < num_neighbours:
            n_val = np.random.normal(cur_value, std)
            if n_val < 0 or n_val > 1:
                continue
            neighbour = self.copy()
            # pylint: disable=protected-access
            neighbour.value = neighbour._normalization_inv(n_val)
            neighbours.append(neighbour)

        return neighbours

    def _normalize_value(self, value):
        if value != value:
            raise ValueError("Float parameter value is NaN!")

        if value is not None:
            low, up, _ = self._get_low_high_default()
            value = np.log(value) if self.log else value
            return (value - low) / (up - low)

    def _normalization_inv(self, value):
        if value != value:
            raise ValueError("Float parameter value is NaN!")

        if value is not None:
            low, up, _ = self._get_low_high_default()
            value = value * (up - low) + low
            return np.exp(value) if self.log else value

    def normalized(self):
        hp = super().normalized()
        hp.value = self._normalize_value(self.value)
        hp.default = self._normalize_value(self.default)
        hp.low = 0
        hp.high = 1
        hp.log = False
        return hp
