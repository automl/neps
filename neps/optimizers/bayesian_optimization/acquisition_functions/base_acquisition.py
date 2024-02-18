from abc import ABC, abstractmethod


class BaseAcquisition(ABC):
    def __init__(self):
        self.surrogate_model = None

    @abstractmethod
    def eval(self, x, asscalar: bool = False):
        """Evaluate the acquisition function at point x2."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def set_state(self, surrogate_model, **kwargs):  # pylint: disable=unused-argument
        self.surrogate_model = surrogate_model
