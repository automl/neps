from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from botorch.acquisition import AcquisitionFunction
from torch import Tensor

if TYPE_CHECKING:
    from botorch.models.gpytorch import GPyTorchModel

    from scalefm.optimizers.models.gp import LogCost


class CostWeightedModifiedJES(AcquisitionFunction):
    """Modified JES with normalized log-cost weighting (GIBBON-style).

    alpha(λ) = IG(λ) / c_norm(λ)

    IG(λ) = 0.5 * [log σ²(λ) - (1/S) Σ_s log σ²_s(λ)]

    c_norm(λ) = max((log C(λ) - log C_min) / (log C_max - log C_min), ε)

    with `log C` the exact `log10` cost of `log_cost` and fixed bounds `C_min`,
    `C_max`, so the weighting does not move with the costs observed so far. `C_min`
    has to be the cheapest config sampled from, every config at or below it gets the
    same, largest weight `1 / ε`.

    Fantasy optima (λ*_s, y*_s) are sampled at C_target by restricting
    get_optimal_samples to candidates near C_target before instantiation.
    """

    def __init__(
        self,
        model: GPyTorchModel,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        log_cost: LogCost | None,
        c_min_log: float,
        c_max_log: float,
        epsilon: float = 1e-4,
    ) -> None:
        """Initialize the acquisition.

        Args:
            model: The GP of the objective.
            optimal_inputs: The sampled optima `(S, D)`.
            optimal_outputs: Their sampled values `(S, 1)`.
            log_cost: The `log10` cost of encoded configs, `None` for no cost
                weighting.
            c_min_log: The `log10` of the smallest cost, `C_min`.
            c_max_log: The `log10` of the largest cost, `C_max`.
            epsilon: Lower bound of `c_norm`.
        """
        super().__init__(model=model)  # type: ignore[arg-type]
        self.optimal_inputs = optimal_inputs
        self.optimal_outputs = optimal_outputs
        self.log_cost = log_cost
        self.c_min_log = c_min_log
        self.c_max_log = c_max_log
        self.epsilon = epsilon

        # Pre-compute fantasy models once so forward() stays fast.
        S = optimal_inputs.shape[0]
        self.fantasy_models: list[GPyTorchModel] = []
        for s in range(S):
            x_s = optimal_inputs[s : s + 1]  # (1, D)
            y_s = optimal_outputs[s : s + 1]  # (1, 1)
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
        ig = ig.mean(dim=-2).squeeze(-1).clamp(min=0.0)  # (batch,)

        # Normalized log-cost weighting
        if self.log_cost is None or abs(self.c_max_log - self.c_min_log) < 1e-8:
            return ig

        # Without gradient, the cost is a black box of the config
        x_first = X[..., 0, :]  # (batch, D)
        log_c = self.log_cost.evaluate(x_first.reshape(-1, x_first.shape[-1]))
        log_c = log_c.view(x_first.shape[:-1])  # (batch,)

        c_norm = ((log_c - self.c_min_log) / (self.c_max_log - self.c_min_log)).clamp(
            min=self.epsilon
        )

        return ig / c_norm
