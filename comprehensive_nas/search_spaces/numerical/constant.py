from typing import Union

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

    def mutate(
        self,
        parent=None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "local_search",
    ):

        child = self.__copy__()
        child.sample()

        return child

    def crossover(self, parent1, parent2=None):
        return self.__copy__().sample(), self.__copy__().sample()

    def _get_neighbours(self, **kwargs):
        pass

    def _transform(self):
        pass

    def _inv_transform(self):
        pass

    def get_dictionary(self):
        return {self.name: self.value}

    def create_from_id(self, identifier):
        self.value = identifier

