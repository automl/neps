from typing import Union

from .numerical import NumericalParameter


class ConstantParameter(NumericalParameter):
    def __init__(self, value: Union[float, int, str]):
        super().__init__()
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"Constant, value: {self.id}"

    def __copy__(self):
        return self.__class__(value=self.value)

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

    def serialize(self):
        return self.value

    def load_from(self, value):
        self.value = value
