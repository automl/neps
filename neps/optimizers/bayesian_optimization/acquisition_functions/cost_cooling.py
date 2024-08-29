from __future__ import annotations

from typing import TYPE_CHECKING

from botorch.acquisition.logei import partial

from neps.optimizers.bayesian_optimization.acquisition_functions.weighted_acquisition import (
    WeightedAcquisition,
)

if TYPE_CHECKING:
    import torch
    from botorch.acquisition import AcquisitionFunction
    from botorch.models.gp_regression import Likelihood
    from botorch.models.model import Model
    from torch import Tensor


def apply_cost_cooling(
    acq_values: Tensor,
    X: Tensor,
    acq: AcquisitionFunction,
    cost_model: Model,
    likelihood: Likelihood,
    alpha: float,
) -> Tensor:
    posterior = likelihood(cost_model(X))
    cost = posterior.mean

    if acq._log:
        # can derive from eq log(x) = log(acq / cost^alpha)
        return acq_values - alpha * cost.log()
    return acq_values / cost.pow(alpha)


def cost_cooled_acq(
    acq_fn: AcquisitionFunction,
    model: Model,
    likelihood: Likelihood,
    used_budget_percentage: float,
    X_pending: torch.Tensor | None = None,
) -> WeightedAcquisition:
    assert 0 <= used_budget_percentage <= 1
    return WeightedAcquisition(
        acq=acq_fn,
        apply_weight=partial(
            apply_cost_cooling,
            cost_model=model,
            likelihood=likelihood,
            alpha=1 - used_budget_percentage,
        ),
        X_pending=X_pending,
    )
