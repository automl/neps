from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
import torch
from botorch.acquisition import MCAcquisitionFunction

from neps.optimizers.bayesian_optimization.acquisition_functions.base_acquisition import (
    BaseAcquisition,
)

if TYPE_CHECKING:
    from neps.priors import Prior


class PiboAcquisition(MCAcquisitionFunction):
    """Compute a prior weighted acquisition function according to PiBO.

    * https://arxiv.org/pdf/2204.11051
    """

    def __init__(
        self,
        acq_fn: MCAcquisitionFunction,
        prior: Prior,
        beta: float,
        n: float,
    ):
        """Initialize the acquisition function.

        Args:
            acq_fn: The acquisition function to be weighted.
            prior: The prior distribution to be used for weighting.
            beta: The beta parameter for weighting.
            n: The denominator for the beta parameter.
        """
        self._log = self.acq_fn._log
        self.acq_fn = acq_fn

        self.beta = beta
        self.n = n
        self.prior = prior

    @override
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        weight = self.beta / self.n
        acq = self.acq_fn(X)

        # The weight is shown as being applied to the pdf and not the log_pdf
        values = acq * self.prior.prob(X) * weight

        # However, if the base acq function advertises as being log,
        # i.e. self._log, then we should return the log of the values
        return torch.log(values) if self._log else values


class DecayingPriorWeightedAcquisition(BaseAcquisition):
    def __init__(
        self,
        base_acquisition,
        pibo_beta=10,
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
        **base_acquisition_kwargs,
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

    def set_state(self, surrogate_model, **kwargs):
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
