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

    def fit_on_model(self, surrogate_model):  # TODO: make set_state
        self.surrogate_model = surrogate_model
