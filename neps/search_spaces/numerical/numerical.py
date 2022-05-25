from abc import abstractmethod

import torch

from ..parameter import HpTensorShape, Parameter


class NumericalParameter(Parameter):
    def __init__(self, value=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    @property
    def id(self):
        return self.value

    @abstractmethod
    def _get_neighbours(self):
        raise NotImplementedError

    def serialize(self):
        return self.value

    def load_from(self, value):
        self.value = value

    @staticmethod
    def get_tensor_shape(hp_instances):  # pylint: disable=unused-argument
        return HpTensorShape(1)

    def get_tensor_value(self, tensor_shape):  # pylint: disable=unused-argument
        return torch.tensor(self.normalized().value, dtype=torch.get_default_dtype())
