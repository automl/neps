from abc import ABC, abstractmethod


class BaseAcquisition(ABC):
    def __init__(self, surrogate_model):
        self.surrogate_model = surrogate_model

        # Storage for the current evaluation on the acquisition function
        self.next_location = None
        self.next_acq_value = None

    def propose_location(self, *args):
        """Propose new locations for subsequent sampling
        This method should be overriden by respective acquisition function implementations."""
        raise NotImplementedError

    def optimize(self):
        """This is the method that user should call for the Bayesian optimisation main loop."""
        raise NotImplementedError

    @abstractmethod
    def eval(self, x, asscalar: bool = False):
        """Evaluate the acquisition function at point x2. This should be overridden by respective acquisition
        function implementations"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def reset_surrogate_model(self, surrogate_model):
        self.surrogate_model = surrogate_model
