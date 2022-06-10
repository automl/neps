import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseAcquisition(ABC):
    def __init__(self):
        self.surrogate_model = None
        self.train_x = None
        self.train_y = None
        self.train_x_tensor = None
        self.train_y_tensor = None
        self.cost_model = None

    @abstractmethod
    def eval(self, x):
        """Evaluate the acquisition function at point x2."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def set_state(
        self, surrogate_model, cost_model=None, **kwargs
    ):  # pylint: disable=unused-argument
        if kwargs:
            logger.warn(
                f"Unused args for the acquisition function {self.__class__.__name__}"
                f" ({kwargs}), maybe you meant to use another acquisition function"
            )

        self.surrogate_model = surrogate_model
        self.cost_model = cost_model
        train_configs, normalized_input = surrogate_model.fitted_on
        # Configurations and list of outputs
        self.train_x, self.train_y = train_configs
        # Normalized train input and output tensors
        self.train_x_tensor, self.train_y_tensor = normalized_input
