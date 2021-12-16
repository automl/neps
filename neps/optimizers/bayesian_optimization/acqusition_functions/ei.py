from typing import Iterable, Tuple, Union

import numpy as np

try:
    import torch
    from torch.distributions import Normal
except ModuleNotFoundError as e:
    from neps.utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message) from e

from .base_acqusition import BaseAcquisition


class ComprehensiveExpectedImprovement(BaseAcquisition):
    def __init__(
        self,
        surrogate_model,
        augmented_ei: bool = False,
        xi: float = 0.0,
        in_fill: str = "best",
        log_ei: bool = False,
        compute_fast: bool = True,
    ):
        """
        This is the graph BO version of the expected improvement
        key differences are:
        1. The input x2 is a networkx graph instead of a vectorial input
        2. the search space (a collection of x1_graphs) is discrete, so there is no gradient-based optimisation. Instead,
        we compute the EI at all candidate points and empirically select the best position during optimisation

        Args:
            surrogate_model: surrogate model, e.g., GP.
            augmented_ei (bool, optional): Using the Augmented EI heuristic modification to the standard expected improvement algorithm
            according to Huang (2006). Defaults to False.
            xi (float, optional): manual exploration-exploitation trade-off parameter.. Defaults to 0.0.
            in_fill (str, optional): the criterion to be used for in-fill for the determination of mu_star. 'best' means the empirical
            best observation so far (but could be susceptible to noise), 'posterior' means the best *posterior GP mean*
            encountered so far, and is recommended for optimization of more noisy functions. Defaults to "best".
            log_ei (bool, optional): log-EI if true otherwise usual EI. Defaults to False.
            compute_fast (bool, optional): if true use vectorized version. Defaults to True.
        """
        super().__init__(surrogate_model=surrogate_model)

        assert in_fill in ["best", "posterior"]
        self.augmented_ei = augmented_ei
        self.xi = xi
        self.in_fill = in_fill
        self.log_ei = log_ei
        self.incumbent = None
        self.compute_fast = compute_fast

    def eval(
        self, x: Iterable, asscalar: bool = False
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """
        Return the negative expected improvement at the query point x2
        """
        try:
            mu, cov = self.surrogate_model.predict(x)
        except ValueError:
            return -1.0  # in case of error. return ei of -1
        std = torch.sqrt(torch.diag(cov))
        mu_star = self._get_incumbent()
        gauss = Normal(torch.zeros(1, device=mu.device), torch.ones(1, device=mu.device))
        # u = (mu - mu_star - self.xi) / std
        # ei = std * updf + (mu - mu_star - self.xi) * ucdf
        if self.log_ei:
            # we expect that f_min is in log-space
            f_min = mu_star - self.xi
            v = (f_min - mu) / std
            ei = np.exp(f_min) * gauss.cdf(v) - np.exp(
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
        if isinstance(x, list) and asscalar:
            return ei.detach().numpy()
        if asscalar:
            ei = ei.detach().numpy().item()
        return ei

    def _compute_incumbent(self) -> torch.Tensor:
        """
        Compute and return the incumbent
        """
        if self.in_fill == "best":
            # return torch.max(self.surrogate_model.y_)
            return torch.min(self.surrogate_model.y_)
        else:
            x = self.surrogate_model.x
            mu_train, _ = self.surrogate_model.predict(x)
            # incumbent_idx = torch.argmax(mu_train)
            incumbent_idx = torch.argmin(mu_train)
            return self.surrogate_model.y_[incumbent_idx]

    def _get_incumbent(self):
        """Get incumbent

        Returns:
            float: incumbent
        """
        return self.incumbent if self.incumbent is not None else self._compute_incumbent()

    def propose_location(
        self, candidates: Iterable, top_n: int = 5, return_distinct: bool = True
    ) -> Tuple[Iterable, np.ndarray, np.ndarray]:
        """top_n: return the top n candidates wrt the acquisition function."""
        # selected_idx = [i for i in self.candidate_idx if self.evaluated[i] is False]
        # eis = torch.tensor([self.eval(self.candidates[c]) for c in selected_idx])
        # print(eis)
        self.incumbent = (
            self._compute_incumbent()
        )  # avoid computing inc over and over again
        if return_distinct:
            if self.compute_fast:
                eis = self.eval(candidates, asscalar=True)  # faster
            else:
                eis = np.array([self.eval(c, asscalar=True) for c in candidates])
            eis_, unique_idx = np.unique(eis, return_index=True)
            try:
                i = np.argpartition(eis_, -top_n)[-top_n:]
                indices = np.array([unique_idx[j] for j in i])
            except ValueError:
                eis = torch.tensor([self.eval(c) for c in candidates])
                _, indices = eis.topk(top_n)
        else:
            eis = torch.tensor([self.eval(c) for c in candidates])
            _, indices = eis.topk(top_n)
        xs = [candidates[int(i)] for i in indices]
        self.incumbent = None
        return xs, eis, indices

    def optimize(self):
        raise ValueError(
            "The kernel invoked does not have hyperparameters to optimse over!"
        )
