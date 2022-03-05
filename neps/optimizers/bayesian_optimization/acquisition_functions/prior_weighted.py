from typing import Iterable, Union

import numpy as np
import torch

from .base_acquisition import BaseAcquisition


class DecayingPriorWeightedAcquisition(BaseAcquisition):
    def __init__(
        self,
        surrogate_model,
        base_acquisition,
    ):
        super().__init__(surrogate_model=surrogate_model)
        self.base_acquisition = base_acquisition

    def eval(
        self,
        x: Iterable,
        **base_acquisition_kwargs,
    ) -> Union[np.ndarray, torch.Tensor, float]:
        return self.base_acquisition(x, **base_acquisition_kwargs)

    def update(self, surrogate_model):
        super().update(surrogate_model)
