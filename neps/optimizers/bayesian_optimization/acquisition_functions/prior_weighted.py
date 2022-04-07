from typing import Iterable, Union

import numpy as np
import torch

from .base_acquisition import BaseAcquisition


class DecayingPriorWeightedAcquisition(BaseAcquisition):
    def __init__(
        self, base_acquisition, pibo_beta=10
    ):  # pylint: disable=super-init-not-called
        self.pibo_beta = pibo_beta
        self.base_acquisition = base_acquisition

    def eval(
        self,
        x: Iterable,
        **base_acquisition_kwargs,
    ) -> Union[np.ndarray, torch.Tensor, float]:
        acquisition = self.base_acquisition(x, **base_acquisition_kwargs)
        num_bo_iterations = len(self.base_acquisition.surrogate_model.x)
        for i, candidate in enumerate(x):
            prior_weight = candidate.compute_prior()
            if prior_weight != 1.0:
                acquisition[i] *= np.power(
                    prior_weight + 1e-12, self.pibo_beta / num_bo_iterations
                )
        return acquisition

    def set_state(self, surrogate_model, **kwargs):
        self.base_acquisition.set_state(surrogate_model)
