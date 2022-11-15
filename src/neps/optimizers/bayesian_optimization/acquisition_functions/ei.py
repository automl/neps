from typing import List

import torch
from torch.distributions import Normal

from .base_acquisition import BaseAcquisition


class ComprehensiveExpectedImprovement(BaseAcquisition):
    def __init__(
        self,
        augmented_ei: bool = False,
        xi: float = 0.0,
        in_fill: str = "best",
        log_ei: bool = False,
        optimize_on_max_fidelity: bool = False,
    ):
        """This is the graph BO version of the expected improvement
        key differences are:

        1. The input x2 is a networkx graph instead of a vectorial input

        2. The search space (a collection of x1_graphs) is discrete, so there is no
           gradient-based optimisation. Instead, we compute the EI at all candidate points
           and empirically select the best position during optimisation

        Args:
            augmented_ei: Using the Augmented EI heuristic modification to the standard
                expected improvement algorithm according to Huang (2006).
            xi: manual exploration-exploitation trade-off parameter.
            in_fill: the criterion to be used for in-fill for the determination of mu_star
                'best' means the empirical best observation so far (but could be
                susceptible to noise), 'posterior' means the best *posterior GP mean*
                encountered so far, and is recommended for optimization of more noisy
                functions. Defaults to "best".
            log_ei: log-EI if true otherwise usual EI.
        """
        super().__init__()

        if in_fill not in ["best", "posterior"]:
            raise ValueError(f"Invalid value for in_fill ({in_fill})")
        self.augmented_ei = augmented_ei
        self.xi = xi
        self.in_fill = in_fill
        self.log_ei = log_ei
        self.incumbent = None
        self.optimize_on_max_fidelity = optimize_on_max_fidelity

    def eval(self, x: List) -> torch.Tensor:
        """
        Return the negative expected improvement at the query point x2
        """
        assert self.incumbent is not None, "EI function not fitted on model"
        assert x, "No configurations for evaluation"

        if x[0].has_fidelity and self.optimize_on_max_fidelity:
            x = x.copy()
            for conf in x:
                conf.set_to_max_fidelity()

        mu, covariance_matrix = self.surrogate_model.predict(x, normalized=True)
        cov = torch.diag(covariance_matrix)
        assert all(cov >= 0)
        std = torch.sqrt(cov)
        mu_star = self.incumbent
        gauss = Normal(torch.zeros(1, device=mu.device), torch.ones(1, device=mu.device))
        # u = (mu - mu_star - self.xi) / std
        # ei = std * updf + (mu - mu_star - self.xi) * ucdf
        if self.log_ei:
            # we expect that f_min is in log-space
            f_min = mu_star - self.xi
            v = (f_min - mu) / std
            ei = torch.exp(f_min) * gauss.cdf(v) - torch.exp(0.5 * cov + mu) * gauss.cdf(
                v - std
            )
        else:
            u = (mu_star - mu - self.xi) / std
            ucdf = gauss.cdf(u)
            updf = torch.exp(gauss.log_prob(u))
            ei = std * updf + (mu_star - mu - self.xi) * ucdf
        if self.augmented_ei:
            sigma_n = self.surrogate_model.gp.likelihood.noise
            ei *= 1.0 - torch.sqrt(torch.tensor(sigma_n, device=mu.device)) / torch.sqrt(
                sigma_n + cov
            )
        return ei

    def set_state(self, surrogate_model, **kwargs):
        super().set_state(surrogate_model, **kwargs)

        # Compute incumbent
        if self.in_fill == "best":
            self.incumbent = torch.min(self.train_y_tensor)
        else:
            mu_train = self.surrogate_model.predict_mean(self.train_x, normalized=True)
            incumbent_idx = torch.argmin(mu_train)
            self.incumbent = self.train_y_tensor[incumbent_idx]