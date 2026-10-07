from __future__ import annotations

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.joint_entropy_search import (
    qJointEntropySearch,
)
from botorch.acquisition.utils import get_optimal_samples
from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor


class CostWeightedModifiedJES(AcquisitionFunction):
    """Modified JES with normalized log-cost weighting (GIBBON-style).

    α(λ) = IG(λ) / c_norm(λ)

    IG(λ) = 0.5 * [log σ²(λ) - (1/S) Σ_s log σ²_s(λ)]

    c_norm(λ) = (log C(λ) - log C_min) / (log C_max - log C_min) + ε

    Fantasy optima (λ*_s, y*_s) are sampled at C_target by restricting
    get_optimal_samples to candidates near C_target before instantiation.
    """

    def __init__(
        self,
        model: GPyTorchModel,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        cost_model: GPyTorchModel | None,
        c_min_log: float,
        c_max_log: float,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__(model=model)  # type: ignore[arg-type]
        self.optimal_inputs = optimal_inputs
        self.optimal_outputs = optimal_outputs
        self.cost_model = cost_model
        self.c_min_log = c_min_log
        self.c_max_log = c_max_log
        self.epsilon = epsilon

        # Pre-compute fantasy models once so forward() stays fast.
        S = optimal_inputs.shape[0]
        self.fantasy_models: list[GPyTorchModel] = []
        for s in range(S):
            x_s = optimal_inputs[s : s + 1]   # (1, D)
            y_s = optimal_outputs[s : s + 1]   # (1, 1)
            self.fantasy_models.append(model.get_fantasy_model(x_s, y_s))  # type: ignore[arg-type]

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the acquisition function.

        Args:
            X: (batch, q, D) — typically q=1 during joint optimization.

        Returns:
            Tensor of shape (batch,).
        """
        # Marginal predictive variance at X: (batch, q, 1)
        marginal_var: Tensor = self.model.posterior(X).variance  # type: ignore[union-attr]
        log_marginal_var = marginal_var.clamp(min=1e-12).log()  # (batch, q, 1)

        # Conditional predictive variance after each fantasy observation
        cond_log_vars = []
        for fm in self.fantasy_models:
            cond_var: Tensor = fm.posterior(X).variance  # type: ignore[union-attr]
            cond_log_vars.append(cond_var.clamp(min=1e-12).log())

        # IG = 0.5 * (log σ² - mean_s log σ²_s), averaged over q, squeezed to (batch,)
        cond_log_var_mean = torch.stack(cond_log_vars, dim=0).mean(dim=0)  # (batch, q, 1)
        ig = 0.5 * (log_marginal_var - cond_log_var_mean)  # (batch, q, 1)
        ig = ig.mean(dim=-2).squeeze(-1).clamp(min=0.0)    # (batch,)

        # Normalized log-cost weighting
        if self.cost_model is None or abs(self.c_max_log - self.c_min_log) < 1e-8:
            return ig

        with torch.no_grad():
            pred_log_c: Tensor = self.cost_model.posterior(X[..., :1, :]).mean  # type: ignore[union-attr]
        pred_log_c = pred_log_c.squeeze(-1).squeeze(-1)  # (batch,)

        c_norm = (
            (pred_log_c - self.c_min_log) / (self.c_max_log - self.c_min_log)
        ).clamp(min=self.epsilon)

        return ig / c_norm
