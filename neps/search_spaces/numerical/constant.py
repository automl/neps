from typing import Union

from .numerical import NumericalParameter


class ConstantParameter(NumericalParameter):
    def __init__(self, value: Union[float, int, str], is_fidelity: bool = False):
        super().__init__()
        self.value = value
        self.is_fidelity = is_fidelity

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.value == other.value

    def __repr__(self):
        return f"Constant, value: {self.id}"

    def sample(self):
        pass

    def mutate(  # pylint: disable=unused-argument
        self,
        parent=None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "local_search",
    ):
        return self

    def crossover(self, parent1, parent2=None):  # pylint: disable=unused-argument
        return self, self

    def _get_neighbours(self, **kwargs):
        pass
