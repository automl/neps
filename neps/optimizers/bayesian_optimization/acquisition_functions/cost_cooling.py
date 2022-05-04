from typing import Iterable, Union

import numpy as np
import torch

from .base_acquisition import BaseAcquisition
from .ei import ComprehensiveExpectedImprovement


class CostCooler(BaseAcquisition):
    def __init__(
        self,
        base_acquisition: BaseAcquisition = ComprehensiveExpectedImprovement,
        **base_acquisition_kwargs,
    ):  # pylint: disable=super-init-not-called
        self.acquisition_function = base_acquisition(**base_acquisition_kwargs)
        self.cost_model = None
        self.alpha = None

    def eval(
        self,
        x: Iterable,
        **base_acquisition_kwargs,
    ) -> Union[np.ndarray, torch.Tensor, float]:
        base_acquisition_value = self.base_acquisition.eval(x, **base_acquisition_kwargs)
        costs = self.cost_model.predict(x)
        return base_acquisition_value / (costs ^ self.alpha)

    def set_state(self, surrogate_model, **kwargs):
        super().set_state(surrogate_model=surrogate_model)
        self.alpha = kwargs.alpha
        self.cost_model = kwargs.cost_model
