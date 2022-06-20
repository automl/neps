from typing import Union

from .numerical import NumericalParameter


class ConstantParameter(NumericalParameter):
    def __init__(self, value: Union[float, int, str], **kwargs):
        super().__init__(value=value, **kwargs)

    def __repr__(self):
        return f"<Constant, value: {self.id}>"

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
        raise NotImplementedError

    def _get_neighbours(self, **kwargs):
        pass
