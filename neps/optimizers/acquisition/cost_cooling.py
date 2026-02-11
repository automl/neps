from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from botorch.acquisition.logei import partial

from neps.optimizers.acquisition.weighted_acquisition import WeightedAcquisition

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction
    from botorch.acquisition.analytic import GPyTorchModel
    from torch import Tensor

logger = logging.getLogger(__name__)


def apply_cost_cooling(
    acq_values: Tensor,
    X: Tensor,
    acq: AcquisitionFunction,
    cost_model: GPyTorchModel,
    alpha: float,
    cost_in_log_scale: bool = True,
) -> Tensor:
    """Apply cost cooling to acquisition values.
    
    The formula is: weighted_acq = acq / cost^alpha
    For log-scale acquisitions: log(weighted_acq) = log(acq) - alpha * log(cost)
    
    Args:
        acq_values: Raw acquisition function values.
        X: Input points.
        acq: The acquisition function (used to check if it's log-scaled).
        cost_model: GP model for predicting costs.
        alpha: Cost cooling exponent (1 - budget_used_percentage).
        cost_in_log_scale: For non-log acquisitions, whether to use log(cost)^alpha 
            (True, stable for large costs 1e8+) or cost^alpha (False, better 
            differentiation for small costs). Has no effect for log acquisitions.
    """
    # Get posterior mean
    posterior = cost_model.posterior(X)
    cost = posterior.mean
    cost = cost.squeeze(dim=-1) if cost_model.num_outputs == 1 else cost.sum(dim=-1)
    
    # Untransform to get costs in original scale (handles Log + Standardize or just Standardize)
    if hasattr(cost_model, 'outcome_transform') and cost_model.outcome_transform is not None:
        cost_for_untransform = cost.unsqueeze(-1)
        dummy_var = torch.ones_like(cost_for_untransform)
        cost_untransformed, _ = cost_model.outcome_transform.untransform(
            cost_for_untransform, dummy_var
        )
        cost = cost_untransformed.squeeze(-1)
    
    # To avoid very small or negative costs in log or power transformations
    cost = cost.clamp(min=1.0)

    if acq._log:
        # Log-scale acquisition (e.g., qLogNoisyExpectedImprovement)
        # log(weighted) = log(acq) - alpha * log(cost)
        w = alpha * cost.log()
        return acq_values - w

    # Non-log acquisition (e.g., qJES)
    if cost_in_log_scale:
        # Use log(cost)^alpha for numerical stability with large costs
        # Compresses 1e8-1e12 → ~18-28
        w = cost.log().clamp(min=1.0).pow(alpha)
    else:
        # Use cost^alpha - better differentiation for small costs
        # but unstable for large costs (1e6+)
        w = cost.pow(alpha)
    
    return torch.where(acq_values > 0, acq_values / w, acq_values * w)


def cost_cooled_acq(
    acq_fn: AcquisitionFunction,
    model: GPyTorchModel,
    used_max_cost_total_percentage: float,
    cost_in_log_scale: bool = True,
) -> WeightedAcquisition:
    """Create a cost-cooled acquisition function.
    
    Args:
        acq_fn: Base acquisition function to wrap.
        model: Cost GP model (can have any outcome transform - we always untransform).
        used_max_cost_total_percentage: Fraction of budget used (0-1).
            alpha = 1 - this value, so early on alpha≈1 (strong cost penalty),
            later alpha≈0 (focus on quality).
        cost_in_log_scale: For non-log acquisitions (e.g., qJES), whether to divide
            by log(cost)^alpha (True, default, stable for large costs) or 
            cost^alpha (False, better for small costs <1e6).
    """
    assert 0 <= used_max_cost_total_percentage <= 1
    return WeightedAcquisition(
        acq=acq_fn,
        apply_weight=partial(
            apply_cost_cooling,
            cost_model=model,
            alpha=1 - used_max_cost_total_percentage,
            cost_in_log_scale=cost_in_log_scale,
        ),
    )
