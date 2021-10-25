from typing import List, Union

from ..hyperparameter import Hyperparameter


class ConstantHyperparameter(Hyperparameter):
    def __init__(self, name: str, value: Union[float, int, str]):
        super(ConstantHyperparameter, self).__init__(name)
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((self.name, self.value))

    def __repr__(self):
        return "Constant {}, value: {}".format(self.name, self.value)

    def __copy__(self):
        return self.__class__(name=self.name, value=self.value)

    def sample(self, seed):
        return self.value

    def mutate(self, parent=None):
        return self.value

    def crossover(self, parent1, parent2=None):
        return self.value
