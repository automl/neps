from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy

import torch


class HpTensorShape:
    def __init__(self, length):
        self.length = length
        self.bounds = None  # First included, last excluded

    def set_bounds(self, begin):
        assert self.bounds is None
        self.bounds = (begin, begin + self.length)

    @property
    def begin(self):
        assert self.bounds is not None
        return self.bounds[0]

    @property
    def end(self):
        assert self.bounds is not None
        return self.bounds[1]

    @property
    def active_dims(self):
        return list(range(self.begin, self.end))


class Parameter:
    def __init__(self, is_fidelity=False, set_default_value=True):
        self.is_fidelity = is_fidelity
        if set_default_value:
            self.value = None

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def mutate(self, parent=None):
        raise NotImplementedError

    @abstractmethod
    def crossover(self, parent1, parent2=None):
        raise NotImplementedError

    @abstractmethod
    def serialize(self):
        raise NotImplementedError

    @abstractmethod
    def load_from(self, data):
        raise NotImplementedError

    def normalized(self):
        return deepcopy(self)

    def compute_prior(self):  # pylint: disable=no-self-use
        return 1

    @staticmethod
    @abstractmethod
    def get_tensor_shape(hp_instances: list[Parameter]) -> HpTensorShape:
        raise NotImplementedError

    @abstractmethod
    def get_tensor_value(self, tensor_shape: HpTensorShape) -> torch.Tensor:
        raise NotImplementedError

    def __eq__(self, other):
        # Assuming that two different classes should represent two different parameters
        return isinstance(other, self.__class__) and self.serialize() == other.serialize()
