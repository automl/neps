from __future__ import annotations

import math
from copy import deepcopy
from typing import Literal

import numpy as np
import scipy.stats

from .numerical import NumericalParameter

FLOAT_CONFIDENCE_SCORES = {
    "low": 0.5,
    "medium": 0.25,
    "high": 0.125,
}


class FloatParameter(NumericalParameter):
    """A float value for a parameter.

    This kind of [`Parameter`][neps.search_spaces.parameter] is used
    to represent hyperparameters with continuous float values, optionally specifying if it exists
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

    def __init__(
        self,
        lower: float | int,
        upper: float | int,
        *,
        log: bool = False,
        is_fidelity: bool = False,
        default: None | float | int = None,
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        """Create a new `FloatParameter`.

        Args:
            lower: lower bound for the hyperparameter.
            upper: upper bound for the hyperparameter.
            log: whether the hyperparameter is on a log scale.
            is_fidelity: whether the hyperparameter is fidelity.
            default: default value for the hyperparameter.
            default_confidence: confidence score for the default value, used when
                condsider prior based optimization.
        """
        super().__init__(is_fidelity=is_fidelity)

        self.default = default
        self.default_confidence_score = FLOAT_CONFIDENCE_SCORES[default_confidence]
        self.has_prior = self.default is not None

        self.lower = float(lower)
        self.upper = float(upper)

        if self.lower >= self.upper:
            raise ValueError(
                f"Float parameter: bounds error (lower >= upper). Actual values: "
                f"lower={self.lower}, upper={self.upper}"
            )

        if self.default is not None:
            if not self.lower <= self.default <= self.upper:
                raise ValueError(
                    f"Float parameter: default bounds error. Expected lower <= default"
                    f" <= upper, but got lower={self.lower}, default={self.default},"
                    f" upper={self.upper}"
                )

        # Validate 'log' and 'is_fidelity' types to prevent configuration errors
        # from the YAML input
        for param, value in {"log": log, "is_fidelity": is_fidelity}.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected '{param}' to be a boolean, but got type: "
                    f"{type(value).__name__}"
                )

        self.log = log

        if self.log:
            self._set_log_values()

        self.value: None | float = None

    def __eq__(self, other):
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

    def __repr__(self):
        float_repr = f"{self.value:.07f}" if self.value is not None else "None"
        return f"<Float, range: [{self.lower}, {self.upper}], value: {float_repr}>"

    def _set_log_values(self):
        if self.lower <= 0:
            raise ValueError("Float parameter: bounds error (log scale).")
        self._lower = np.log(self.lower)
        self._upper = np.log(self.upper)
        if self.default is not None:
            self._default = np.log(self.default)
        else:
            self._default = None

    def _get_low_high_default(self):
        if self.log:
            self._set_log_values()
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
        mutation_rate: float = 1.0,
        mutation_strategy: str = "local_search",
        **kwargs,
    ):
        if self.is_fidelity:
            raise ValueError("Trying to mutate fidelity param!")
        if parent is None:
            parent = self

        if mutation_strategy == "simple":
            child = deepcopy(self)
            child.sample()
        elif mutation_strategy == "local_search":
            if "std" in kwargs:
                child = self._get_neighbours(std=kwargs["std"], num_neighbours=1)[
                    0
                ]
            else:
                child = self._get_neighbours(num_neighbours=1)[
                    0
                ]
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

        proxy_self = deepcopy(self)
        proxy_self.value = (parent1.value + parent2.value) / 2
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
            neighbour = deepcopy(self)
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

    def set_default_confidence_score(self, default_confidence):
        self.default_confidence_score = FLOAT_CONFIDENCE_SCORES[default_confidence]
