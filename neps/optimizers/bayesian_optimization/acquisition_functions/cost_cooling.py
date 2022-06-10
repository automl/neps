from typing import Iterable, Union

import numpy as np
import torch

from ..default_consts import EPSILON
from .base_acquisition import BaseAcquisition


class CostCooler(BaseAcquisition):
    def __init__(
        self,
        base_acquisition: BaseAcquisition,
    ):  # pylint: disable=super-init-not-called
        self.base_acquisition = base_acquisition
        self.cost_model = None
        self.alpha = None

    def eval(self, x: Iterable) -> Union[np.ndarray, torch.Tensor, float]:
        base_acquisition_value = self.base_acquisition.eval(x)
        with torch.no_grad():
            costs, _ = self.cost_model.predict(x)
            costs = costs.detach().numpy()
        return base_acquisition_value / np.maximum(costs**self.alpha, EPSILON)
        # return base_acquisition_value / np.maximum(costs**self.alpha, EPSILON)

    def set_state(
        self, surrogate_model, alpha, cost_model, update_base_model=True, **kwargs
    ):
        super().set_state(surrogate_model=surrogate_model, cost_model=cost_model)
        if update_base_model:
            self.base_acquisition.set_state(
                surrogate_model, cost_model=cost_model, **kwargs
            )
        self.alpha = alpha
