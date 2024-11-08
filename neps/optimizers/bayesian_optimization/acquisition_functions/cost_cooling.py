from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from botorch.acquisition.logei import partial

from neps.optimizers.bayesian_optimization.acquisition_functions.weighted_acquisition import (  # noqa: E501
    WeightedAcquisition,
)

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction
    from botorch.acquisition.analytic import GPyTorchModel
    from torch import Tensor


def apply_cost_cooling(
    acq_values: Tensor,
    X: Tensor,
    acq: AcquisitionFunction,
    cost_model: GPyTorchModel,
    alpha: float,
) -> Tensor:
    # NOTE: We expect **positive** costs from model
    cost = cost_model.posterior(X).mean
    cost = cost.squeeze(dim=-1) if cost_model.num_outputs == 1 else cost.sum(dim=-1)

    if acq._log:
        # Take log of both sides, acq is already log scaled
        # -- x = acq / cost^alpha
        # -- log(x) = log(acq) - alpha * log(cost)
        w = alpha * cost.log()
        return acq_values - w

    # https://github.com/pytorch/botorch/discussions/2194
    w = cost.pow(alpha)
    return torch.where(acq_values > 0, acq_values / w, acq_values * w)


def cost_cooled_acq(
    acq_fn: AcquisitionFunction,
    model: GPyTorchModel,
    used_max_cost_total_percentage: float,
) -> WeightedAcquisition:
    assert 0 <= used_max_cost_total_percentage <= 1
    return WeightedAcquisition(
        acq=acq_fn,
        apply_weight=partial(
            apply_cost_cooling,
            cost_model=model,
            alpha=1 - used_max_cost_total_percentage,
        ),
    )
