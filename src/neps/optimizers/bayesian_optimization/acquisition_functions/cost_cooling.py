from typing import Iterable

import torch

# from ..default_consts import EPSILON
from .base_acquisition import BaseAcquisition


class CostCooler(BaseAcquisition):
    def __init__(
        self,
        base_acquisition: BaseAcquisition,
    ):  # pylint: disable=super-init-not-called
        super().__init__()
        self.base_acquisition = base_acquisition
        self.cost_model = None
        self.alpha = None

    def eval(self, x: Iterable) -> torch.Tensor:
        base_acquisition_value = self.base_acquisition.eval(x)
        costs, _ = self.cost_model.predict(x)
        # if costs < 0.001:
        #     costs = 1
        if torch.is_tensor(costs):
            cost_cooled = torch.zeros_like(costs)
            index = 0
            for _, y in enumerate(costs.detach().numpy()):
                if y < 0.0001:
                    cost_cooled[index] = base_acquisition_value[index]
                else:
                    cost_cooled[index] = base_acquisition_value[index] / (y**self.alpha)
                index += 1
        # return base_acquisition_value # / (costs**self.alpha).detach().numpy()
        return cost_cooled

    def set_state(
        self, surrogate_model, alpha, cost_model, update_base_model=True, **kwargs
    ):
        super().set_state(surrogate_model=surrogate_model, cost_model=cost_model)
        if update_base_model:
            self.base_acquisition.set_state(
                surrogate_model, cost_model=cost_model, **kwargs
            )
        self.alpha = alpha