from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from .base_acquisition import BaseAcquisition


class UpperConfidenceBound(BaseAcquisition):
    def __init__(self, beta: float = 1.0, maximize: bool = False):
        """Upper Confidence Bound (UCB) acquisition function.

        Args:
            beta: Controls the balance between exploration and exploitation.
            maximize: If True, maximize the given model, else minimize.
                DEFAULT=False, assumes minimzation.
        """
        super().__init__()
        self.beta = beta  # can be updated as part of the state for dynamism or a schedule
        self.maximize = maximize

        # to be initialized as part of the state
        self.surrogate_model = None

    def set_state(self, surrogate_model, **kwargs):
        super().set_state(surrogate_model)
        self.surrogate_model = surrogate_model
        if "beta" in kwargs:
            if not isinstance(kwargs["beta"], list | np.array):
                self.beta = kwargs["beta"]
            else:
                self.logger.warning("Beta is a list, not updating beta value!")

    def eval(
        self, x: Iterable, asscalar: bool = False
    ) -> np.ndarray | torch.Tensor | float:
        try:
            mu, cov = self.surrogate_model.predict(x)
            std = torch.sqrt(torch.diag(cov))
        except ValueError as e:
            raise e
        sign = 1 if self.maximize else -1  # LCB is performed if minimize=True
        ucb_scores = mu + sign * np.sqrt(self.beta) * std
        # if LCB, minimize acquisition, or maximize -acquisition
        return ucb_scores.detach().numpy() * sign
