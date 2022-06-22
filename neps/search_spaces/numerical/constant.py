from typing import Union

import torch

from ..parameter import HpTensorShape
from .numerical import NumericalParameter


class ConstantParameter(NumericalParameter):
    def __init__(self, value: Union[float, int, str], **kwargs):
        super().__init__(value=value, **kwargs)

    def __repr__(self):
        return f"<Constant, value: {self.id}>"

    def sample(self, user_priors: bool = False):
        pass

    def prior_probability(self):  # pylint: disable=no-self-use
        return 1.0

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

    @staticmethod
    def get_tensor_shape(hp_instances):
        return HpTensorShape(0, hp_instances)

    def get_tensor_value(
        self, tensor_shape
    ):  # pylint: disable=unused-argument,no-self-use
        return torch.tensor([])
