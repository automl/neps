from abc import ABC, abstractmethod


class BaseAcquisition(ABC):
    def __init__(self):
        self.surrogate_model = None
        self.train_x = None
        self.train_y = None
        self.train_x_tensor = None
        self.train_y_tensor = None

    @abstractmethod
    def eval(self, x, asscalar: bool = False):
        """Evaluate the acquisition function at point x2."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def set_state(self, surrogate_model, **kwargs):  # pylint: disable=unused-argument
        self.surrogate_model = surrogate_model
        train_configs, normalized_input = surrogate_model.fitted_on
        # Configurations and list of outputs
        self.train_x, self.train_y = train_configs
        # Normalized train input and output tensors
        self.train_x_tensor, self.train_y_tensor = normalized_input
