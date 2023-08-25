from typing import Any, Iterable, Tuple, Union

import numpy as np
import torch
from torch.distributions import Normal

from ...multi_fidelity.utils import MFObservedData
from .base_acquisition import BaseAcquisition
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

    def preprocess(self, x: Iterable) -> Tuple[Iterable, Iterable]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity Expected Improvement acquisition function.
        """
        budget_list = []
        # TODO: get the appropriate mapping of budget to incumbent (a dict?)
        _configs = self.observations.get_incumbents_for_budgets()

        # TODO: check that the samples here have their fidelities appropriately set
        # incrementing budget by b_step for all candidates
        # collecting the list of budgets over which incumbent needs to be found
        for _x in x:
            _x.fidelity.value = _x.fidelity.value + self.b_step  # +1 step in budget
            budget_list.append(_x.fidelity.value)

        # finding the incumbent for each budget
        # TODO: how to do this correctly?
        # this step creates a one-to-one ordering wrt x that assings the relevant
        # incumbent value for it such that it is the best value seen at a budget b+1
        # where b is the maximum steps seen by the config in x
        inc_list = [_configs.budget.perf for budget in budget_list]

        return x, inc_list

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
        self, surrogate_model: Any, observations: MFObservedData, b_step: int, **kwargs
    ):
        # overload to select incumbent differently through observations
        self.surrogate_model = surrogate_model
        self.observations = observations
        self.b_step = b_step
        return
