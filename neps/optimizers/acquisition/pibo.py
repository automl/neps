"""# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
Prior-Guided Acquisition Functions

References:

.. [Hvarfner2022]
    C. Hvarfner, D. Stoll, A. Souza, M. Lindauer, F. Hutter, L. Nardi. PiBO:
    Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization.
    ICLR 2022.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botorch.acquisition.logei import partial

from neps.optimizers.acquisition.weighted_acquisition import WeightedAcquisition

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import AcquisitionFunction
    from torch import Tensor

    from neps.sampling import Prior
    from neps.space import ConfigEncoder, Domain


def apply_pibo_acquisition_weight(
    acq_values: Tensor,
    X: Tensor,
    acq: AcquisitionFunction,
    *,
    prior: Prior,
    x_domain: Domain | list[Domain] | ConfigEncoder,
    prior_exponent: float,
) -> Tensor:
    """Apply piBO weighting to acquisition values.
    
    Note: X may have shape (batch, q, d) where q includes X_pending points
    concatenated by the WeightedAcquisition decorator. The prior returns
    per-candidate values (batch, q), but acq_values is (batch,) after the
    base acquisition has already reduced over q. We only weight based on the
    FIRST candidate (the new one being proposed), not the pending ones.
    """
    # Check if acquisition is log-scale (safe attribute access)
    is_log_acq = getattr(acq, '_log', False)
    
    if is_log_acq:
        # prior.log_pdf returns (batch, q) for X shape (batch, q, d)
        log_probs = prior.log_pdf(X, frm=x_domain)
        
        # Only use the first candidate's prior (not pending points)
        # log_probs shape: (batch, q) or (batch,)
        # Keep as (batch, 1) to match acq_values shape from WeightedAcquisition
        if log_probs.ndim > 1 and log_probs.shape[-1] > 1:
            log_probs = log_probs[..., :1]  # (batch, 1) - keep dim!
        elif log_probs.ndim == 1:
            log_probs = log_probs.unsqueeze(-1)  # (batch,) -> (batch, 1)
            
        weighted_log_probs = log_probs * prior_exponent
        return acq_values + weighted_log_probs

    # Non-log case
    probs = prior.pdf(X, frm=x_domain)
    
    # Only use the first candidate's prior (not pending points)
    # Keep as (batch, 1) to match acq_values shape from WeightedAcquisition
    if probs.ndim > 1 and probs.shape[-1] > 1:
        probs = probs[..., :1]  # (batch, 1) - keep dim!
    elif probs.ndim == 1:
        probs = probs.unsqueeze(-1)  # (batch,) -> (batch, 1)
        
    weighted_probs = probs.pow(prior_exponent)
    return acq_values * weighted_probs


def pibo_acquisition(
    acq_fn: AcquisitionFunction,
    prior: Prior,
    prior_exponent: float,
    x_domain: Domain | list[Domain] | ConfigEncoder,
) -> WeightedAcquisition:
    return WeightedAcquisition(
        acq=acq_fn,
        apply_weight=partial(
            apply_pibo_acquisition_weight,
            prior=prior,
            x_domain=x_domain,
            prior_exponent=prior_exponent,
        ),
    )
