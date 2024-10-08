from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from neps.optimizers.bayesian_optimization.acquisition_functions.base_acquisition import (
    BaseAcquisition,
)

if TYPE_CHECKING:
    import torch


class DecayingPriorWeightedAcquisition(BaseAcquisition):
    def __init__(
        self,
        base_acquisition: BaseAcquisition,
        *,
        pibo_beta: int = 10,
        log: bool = False,
    ):
        super().__init__()
        self.pibo_beta = pibo_beta
        self.base_acquisition = base_acquisition
        self.log = log
        self.decay_t = 0.0

    def eval(
        self,
        x: Iterable,
        **base_acquisition_kwargs: Any,
    ) -> np.ndarray | torch.Tensor | float:
        acquisition = self.base_acquisition(x, **base_acquisition_kwargs)

        if self.log:
            min_acq_val = abs(min(acquisition)) if min(acquisition) < 0 else 0

        for i, candidate in enumerate(x):
            prior_weight = candidate.compute_prior(log=self.log)
            if prior_weight != 1.0:
                if self.log:
                    # for log -> the smaller the prior_weight,
                    # the more unlikely it is from the prior
                    # also shift acquisition values to avoid negativ values
                    acquisition[i] = (
                        np.log(acquisition[i] + min_acq_val + 1e-12)
                        + (self.pibo_beta / self.decay_t) * prior_weight
                    )
                else:
                    acquisition[i] *= np.power(
                        prior_weight + 1e-12, self.pibo_beta / self.decay_t
                    )
        return acquisition

    def set_state(self, surrogate_model: Any, **kwargs: Any) -> None:
        if "decay_t" in kwargs:
            decay_t = kwargs.pop("decay_t")
        else:
            train_x = surrogate_model.x
            if train_x[0].has_fidelity:
                decay_t = np.sum(
                    [float(_x.fidelity.value >= _x.fidelity.upper) for _x in train_x]
                )
            else:
                decay_t = len(train_x)
        self.decay_t = decay_t
        self.base_acquisition.set_state(surrogate_model, **kwargs)
