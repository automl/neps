from typing import List, Union
import numpy as np

from ..hyperparameter import Hyperparameter


class ConstantHyperparameter(Hyperparameter):
    def __init__(self, name: str, value: Union[float, int, str]):
        super().__init__(name)
        self.value = value
        self._id = -1

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((self.name, self._id, self.value))

    def __repr__(self):
        return "Constant {}-{:.07f}, value: {}".format(self.name, self._id, self.value)

    def __copy__(self):
        return self.__class__(name=self.name, value=self.value)

    def sample(self):
        self._id = np.random.random()

    def mutate(self,
               parent=None,
               mutation_rate: float = 1.0,
               mutation_strategy: str = "simple"):

        child = self.__copy__()
        child.sample()

        return child

    def crossover(self, parent1, parent2=None):
        return self.__copy__().sample(), self.__copy__().sample()
