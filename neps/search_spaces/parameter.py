from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy

import torch


class HpTensorShape:
    """Used to describe the tensor representation of a group of values for a given
    hyperparameter. Can be inherited to allow to store values needed to exploit
    this tensor representation using a kernel.

    Attributes:
        length: size of the tensor
        bounds: bounds of the part of the tensor for the whole configuration that
            will be used to represent values from this group only.
        hp_instances: instanced of the hyperparameters that will be represented
            using this shape.
    """

    def __init__(self, length, hp_instances):
        self.length: int = length
        self.bounds: tuple[int, int] | None = None  # First included, last excluded
        self.hp_instances: list[Parameter] = hp_instances

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
    def sample(self, user_priors: bool = False):
        """Should sample a new value in-place."""
        raise NotImplementedError

    @abstractmethod
    def prior_probability(self) -> float:
        """Should return a value proportional to the prior probability of the value.
        The scale of the values doesn't matter."""
        raise NotImplementedError

    @abstractmethod
    def mutate(self, parent=None):
        raise NotImplementedError

    @abstractmethod
    def crossover(self, parent1, parent2=None):
        raise NotImplementedError

    @abstractmethod
    def serialize(self):
        """Should return a representation of the hyperparameter value as
        a JSON-serializable object."""
        raise NotImplementedError

    @abstractmethod
    def load_from(self, data):
        """Should load a value from a serialized representation.

        Args:
            data: a representation of the hyperparameter value as defined in
                the serialize method
        """
        raise NotImplementedError

    def normalized(self):
        """Should send a copy of the hyperparameter in a normalized value, that
        can be exploited directly by a GP kernel or another method."""
        return self.copy()

    def compute_prior(self):  # pylint: disable=no-self-use
        return 1

    @staticmethod
    @abstractmethod
    def get_tensor_shape(hp_instances: list[Parameter]) -> HpTensorShape:
        """Should return an HpTensorShape object representing the shape of a
        group of values for this hyperparameter.

        Args:
            hp_instances: the list of the hyperparameter instances that should
                be represented by the return value."""
        raise NotImplementedError

    @abstractmethod
    def get_tensor_value(self, tensor_shape: HpTensorShape) -> torch.Tensor:
        """Should return a tensor representation of the normalized
            hyperparameter value.

        Args:
            tensor_shape: describes the shape of the tensor. May contain other
                informations as defined in get_tensor_shape.
        """
        raise NotImplementedError

    def copy(self):
        """Should return the shallowest copy of the same space, creating a new
        HP without copying the objects referenced by it."""
        return deepcopy(self)

    def __eq__(self, other):
        # Assuming that two different classes should represent two different parameters
        return isinstance(other, self.__class__) and self.serialize() == other.serialize()
