# type: ignore
from typing import Any, Iterable, Tuple, Union

import numpy as np
import torch
from torch.distributions import Normal

from ...multi_fidelity.utils import MFObservedData
from .ei import ComprehensiveExpectedImprovement


class MFEI(ComprehensiveExpectedImprovement):
    def __init__(
        self,
        augmented_ei: bool = False,
        xi: float = 0.0,
        in_fill: str = "best",
        log_ei: bool = False,
    ):
        super().__init__(augmented_ei, xi, in_fill, log_ei)
        self.surrogate_model = None
        self.observations = None
        self.b_step = None

    def get_budget_level(self, config) -> int:
        return int((config.fidelity.value - config.fidelity.lower) / self.b_step)

    def preprocess(self, x: Iterable) -> Tuple[Iterable, Iterable]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity Expected Improvement acquisition function.
        """
        budget_list = []
        performances = self.observations.get_best_performance_for_each_budget()

        for _x in x:
            budget_list.append(self.get_budget_level(_x))

        inc_list = []
        for budget_level in budget_list:
            if budget_level in performances.index:
                inc = performances[budget_level]
            else:
                inc = self.observations.get_best_seen_performance()
            inc_list.append(inc)

        return x, torch.Tensor(inc_list)

    def eval(
        self, x: Iterable, asscalar: bool = False
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """Vanilla-EI modified to preprocess samples and accept list of incumbents."""
        x, inc_list = self.preprocess(x)  # IMPORTANT change from vanilla-EI

        _x = x.copy()
        try:
            mu, cov = self.surrogate_model.predict(_x)
        except ValueError as e:
            raise e
            # return -1.0  # in case of error. return ei of -1
        std = torch.sqrt(torch.diag(cov))

        mu_star = inc_list  # IMPORTANT change from vanilla-EI

        gauss = Normal(torch.zeros(1, device=mu.device), torch.ones(1, device=mu.device))
        # u = (mu - mu_star - self.xi) / std
        # ei = std * updf + (mu - mu_star - self.xi) * ucdf
        if self.log_ei:
            # we expect that f_min is in log-space
            f_min = mu_star - self.xi
            v = (f_min - mu) / std
            ei = torch.exp(f_min) * gauss.cdf(v) - torch.exp(
                0.5 * torch.diag(cov) + mu
            ) * gauss.cdf(v - std)
        else:
            u = (mu_star - mu - self.xi) / std
            ucdf = gauss.cdf(u)
            updf = torch.exp(gauss.log_prob(u))
            ei = std * updf + (mu_star - mu - self.xi) * ucdf
        if self.augmented_ei:
            sigma_n = self.surrogate_model.likelihood
            ei *= 1.0 - torch.sqrt(torch.tensor(sigma_n, device=mu.device)) / torch.sqrt(
                sigma_n + torch.diag(cov)
            )
        if isinstance(_x, list) and asscalar:
            return ei.detach().numpy()
        if asscalar:
            ei = ei.detach().numpy().item()
        return ei

    def set_state(
        self,
        surrogate_model: Any,
        observations: MFObservedData,
        b_step: Union[int, float],
        **kwargs,
    ):
        # overload to select incumbent differently through observations
        self.surrogate_model = surrogate_model
        self.observations = observations
        self.b_step = b_step
        return
