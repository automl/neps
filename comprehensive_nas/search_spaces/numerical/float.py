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
            raise ValueError(f"Hp {name}: bounds error (lower >= upper).")

        self.log = log

        if self.log:
            if self.lower <= 0:
                raise ValueError(f"Hp {name}: bounds error (log scale).")
            self._lower = np.log(self.lower)
            self._upper = np.log(self.upper)

        self.value = None

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
        return hash((self.name, self.lower, self.upper, self.log, self.value))

    def __repr__(self):
        return "Float {}, range: [{}, {}], value: {:.07f}".format(
            self.name, self.lower, self.upper, self.value
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
        neighbours = []
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
            raise ValueError(f"Hp-{self.name} value is NaN!")

        self.value = (self.value - self.lower) / (self.upper - self.lower)

    def _inv_transform(self):
        if self.value != self.value:
            raise ValueError(f"Hp-{self.name} value is NaN!")

        self.value = self.value * (self.upper - self.lower) + self.lower

    def get_dictionary(self):
        return {self.name: self.value}

    def create_from_id(self, identifier):
        self.value = identifier
