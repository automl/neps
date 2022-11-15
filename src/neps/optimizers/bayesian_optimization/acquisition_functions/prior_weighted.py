from typing import Iterable

import torch

from ..default_consts import EPSILON
from .base_acquisition import BaseAcquisition


class DecayingPriorWeightedAcquisition(BaseAcquisition):
    def __init__(
        self,
        base_acquisition,
        pibo_beta=10,
        log: bool = False,
    ):  # pylint: disable=super-init-not-called
        self.pibo_beta = pibo_beta
        self.base_acquisition = base_acquisition
        self.log = log

    def eval(
        self,
        x: Iterable,
        **base_acquisition_kwargs,
    ) -> torch.Tensor:
        super().__init__()
        acquisition = self.base_acquisition(x, **base_acquisition_kwargs)

        (train_x, _), _ = self.base_acquisition.surrogate_model.fitted_on
        if train_x[0].has_fidelity:
            decay_t = torch.sum(
                _x.fidelity.value >= _x.fidelity.upper + EPSILON for _x in train_x
            )
        else:
            decay_t = len(train_x)

        if self.log:
            min_acq_val = (
                torch.abs(torch.min(acquisition)) if torch.min(acquisition) < 0 else 0
            )
        for i, candidate in enumerate(x):
            prior_weight = candidate.compute_prior(log=self.log)
            if prior_weight != 1.0:
                if self.log:
                    # for log -> the smaller the prior_weight,
                    # the more unlikely it is from the prior
                    # also shift acquisition values to avoid negativ values
                    acquisition[i] = (
                        torch.log(acquisition[i] + min_acq_val + 1e-12)
                        + (self.pibo_beta / decay_t) * prior_weight
                    )
                else:
                    acquisition[i] *= torch.power(
                        prior_weight + 1e-12, self.pibo_beta / decay_t
                    )
        return acquisition

    def set_state(self, surrogate_model, **kwargs):
        self.base_acquisition.set_state(surrogate_model, **kwargs)
