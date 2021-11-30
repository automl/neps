from typing import Union

from ..hyperparameter import Hyperparameter


class ConstantHyperparameter(Hyperparameter):
    def __init__(self, name: str, value: Union[float, int, str]):
        super().__init__(name)
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((self.name, self.value))

    def __repr__(self):
        return f"Constant {self.name}, value: {self._id}"

    def __copy__(self):
        return self.__class__(name=self.name, value=self.value)

    def sample(self):
        pass

    def mutate(  # pylint: disable=unused-argument
        self,
        parent=None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "local_search",
    ):

        child = self.__copy__()
        child.sample()

        return child

    def crossover(self, parent1, parent2=None):  # pylint: disable=unused-argument
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
