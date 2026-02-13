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
        X: Input points. May have shape (batch, q, d) if X_pending was concatenated.
        acq: The acquisition function (used to check if it's log-scaled).
        cost_model: GP model for predicting costs.
        alpha: Cost cooling exponent (1 - budget_used_percentage).
        cost_in_log_scale: For non-log acquisitions, whether to use log(cost)^alpha 
            (True, stable for large costs 1e8+) or cost^alpha (False, better 
            differentiation for small costs). Has no effect for log acquisitions.
    
    Note: When X_pending is present, X has shape (batch, q, d) where q > 1.
        The base acquisition already reduced over q, so acq_values is (batch,).
        We must only use the FIRST candidate's cost prediction, not the pending ones.
    """
    # Handle X_pending: only use first candidate for cost prediction
    # X shape: (batch, d) or (batch, q, d)
    if X.ndim == 3 and X.shape[-2] > 1:
        # Only predict cost for the first (new) candidate, not pending points
        X_for_cost = X[..., :1, :]  # (batch, 1, d)
    else:
        X_for_cost = X
    
    # Get posterior mean - detach from gradient graph for numerical stability
    with torch.no_grad():
        posterior = cost_model.posterior(X_for_cost)
        cost = posterior.mean.clone()
    
    # Squeeze only the output dimension (last dim), keep q dimension
    # posterior.mean shape: (batch, q, 1) -> (batch, q) after squeeze
    cost = cost.squeeze(dim=-1) if cost_model.num_outputs == 1 else cost.sum(dim=-1)
    
    # Take only first candidate's cost if q > 1 (pending points)
    # cost shape: (batch, q) -> (batch, 1) or stays (batch, 1)
    if cost.ndim > 1 and cost.shape[-1] > 1:
        cost = cost[..., :1]  # (batch, 1) - keep the dimension!
    
    if torch.isnan(cost).any() or torch.isinf(cost).any():
        logger.warning("NaN/Inf in predicted costs, replacing with 1.0")
        cost = torch.nan_to_num(cost, nan=1.0, posinf=1e12, neginf=1.0)
    
    # To avoid very small or negative costs in log or power transformations
    cost = cost.clamp(min=1.0)

    # Check if acquisition is log-scale (safe attribute access)
    is_log_acq = getattr(acq, '_log', False)
    
    if is_log_acq:
        # Log-scale acquisition (e.g., qLogNoisyExpectedImprovement)
        # log(weighted) = log(acq) - alpha * log(cost)
        w = alpha * cost.log()
        return acq_values - w

    # Non-log acquisition (e.g., qJES returns information gain in nats, always >= 0)
    if cost_in_log_scale:
        # Use log(cost)^alpha for numerical stability with large costs
        # Compresses 1e8-1e12 → ~18-28
        log_cost = cost.log().clamp(min=1.0)
        w = log_cost.pow(alpha)
    else:
        # Use cost^alpha - better differentiation for small costs
        # but unstable for large costs (1e6+)
        w = cost.pow(alpha)
    
    # Clamp weights to prevent extreme values
    w = w.clamp(min=1e-6, max=1e6)
    
    # Simple division - qJES values should always be non-negative
    return acq_values / w


def cost_cooled_acq(
    acq_fn: AcquisitionFunction,
    model: GPyTorchModel,
    used_max_cost_total_percentage: float,
    cost_in_log_scale: bool = True,
) -> WeightedAcquisition:
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
