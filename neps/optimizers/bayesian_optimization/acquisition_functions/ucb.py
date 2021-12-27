from typing import Iterable, Tuple

import numpy as np
import torch

from .base_acquisition import BaseAcquisition


class ComprehensiveUpperConfidentBound(BaseAcquisition):
    """
    Graph version of the upper confidence bound acquisition function
    """

    def __init__(self, surrogate_model, beta=None, iters=0):
        """Same as graphEI with the difference that a beta coefficient is asked for, as per standard GP-UCB acquisition"""
        super().__init__(surrogate_model=surrogate_model)

        self.beta = beta

    def eval(self, x, asscalar=False):
        # TODO: predict on graph/hps/both/..
        mu, cov = self.surrogate_model.predict(x)
        std = torch.sqrt(torch.diag(cov))
        if self.beta is None:
            self.beta = 3.0 * torch.sqrt(
                0.5 * torch.log(torch.tensor(2.0 * self.iters + 1.0))
            )
        acq = mu + self.beta * std
        if asscalar:
            acq = acq.detach().numpy().item()
        return acq  # .mean()

    def propose_location(
        self, candidates: Iterable, top_n: int = 5, return_distinct: bool = True
    ) -> Tuple[Iterable, np.ndarray, np.ndarray]:
        self.iters += 1
        raise NotImplementedError
