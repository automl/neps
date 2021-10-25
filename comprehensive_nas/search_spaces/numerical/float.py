from typing import Union, List
import numpy as np
import math

from ..hyperparameter import Hyperparameter


class FloatHyperparameter(Hyperparameter):
    def __init__(self, name: str, lower: Union[float, int], upper: Union[float, int],
                 log: bool = False):
        super(FloatHyperparameter, self).__init__(name)

        self.lower = float(lower)
        self.upper = float(upper)

        if self.lower >= self.upper:
            raise ValueError("Hp {}: bounds error (lower >= upper).".format(name))

        self.log = log

        if self.log:
            if self.lower <= 0:
                raise ValueError("Hp {}: bounds error (log scale).".format(name))
            self._lower = np.log(self.lower)
            self._upper = np.log(self.upper)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name and
            self.lower == other.lower and
            self.upper == other.upper and
            self.log == other.log
        )

    def __hash__(self):
        return hash(
            (
                self.name,
                self.lower,
                self.upper,
                self.log
            )
        )

    def __repr__(self):
        return "Float {}, range: [{}, {}]".format(
            self.name, self.lower, self.upper
        )

    def __copy__(self):
        return self.__class__(
            name=self.name,
            lower=self.lower,
            upper=self.upper,
            log=self.log
        )

    def sample(self, random_state: np.random):
        if self.log:
            value = random_state.uniform(low=self._lower, high=self._upper)
            value = math.exp(value)
        else:
            value = random_state.uniform(low=self.lower, high=self.upper)
        return min(self.upper, max(self.lower, value))

    def mutate(self, parent=None):
        raise NotImplementedError

    def crossover(self, parent1, parent2=None):
        raise NotImplementedError

