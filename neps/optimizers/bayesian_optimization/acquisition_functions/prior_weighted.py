from typing import Iterable, Union

import numpy as np
import torch

from .base_acquisition import BaseAcquisition


class DecayingPriorWeightedAcquisition(BaseAcquisition):
    def __init__(self, base_acquisition):  # pylint: disable=super-init-not-called
        self.base_acquisition = base_acquisition

    def eval(
        self,
        x: Iterable,
        **base_acquisition_kwargs,
    ) -> Union[np.ndarray, torch.Tensor, float]:
        acquisition = self.base_acquisition(x, **base_acquisition_kwargs)
        for i, candidate in enumerate(x):
            prior_weight = candidate.compute_prior()
            acquisition[i] *= prior_weight
        return acquisition

    def update(self, surrogate_model):
        self.base_acquisition.update(surrogate_model)
