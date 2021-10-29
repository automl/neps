import math
from typing import Union

import numpy as np

from ..hyperparameter import Hyperparameter


class FloatHyperparameter(Hyperparameter):
    def __init__(
        self,
        name: str,
        lower: Union[float, int],
        upper: Union[float, int],
        log: bool = False,
    ):
        super().__init__(name)

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

        self.value = None
        self._id = -1

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.lower == other.lower
            and self.upper == other.upper
            and self.log == other.log
            and self.value == other.value
        )

    def __hash__(self):
        return hash((self.name, self._id, self.lower, self.upper, self.log, self.value))

    def __repr__(self):
        return "Float {}-{:.07f}, range: [{}, {}], value: {:.07f}".format(
            self.name, self._id, self.lower, self.upper, self.value
        )

    def __copy__(self):
        return self.__class__(
            name=self.name, lower=self.lower, upper=self.upper, log=self.log
        )

    def sample(self):
        if self.log:
            value = np.random.uniform(low=self._lower, high=self._upper)
            value = math.exp(value)
        else:
            value = np.random.uniform(low=self.lower, high=self.upper)
        self.value = min(self.upper, max(self.lower, value))
        self._id = np.random.random()

    def mutate(
        self, parent=None, mutation_rate: float = 1.0, mutation_strategy: str = "simple"
    ):

        if parent is None:
            parent = self

        if mutation_strategy == "simple":
            child = self.__copy__()
            child.sample()
        else:
            raise NotImplementedError

        if parent.value == child.value:
            raise ValueError("Parent is the same as child!")

        return child

    def crossover(self, parent1, parent2=None):
        raise NotImplementedError
