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

from neps.optimizers.bayesian_optimization.acquisition_functions.weighted_acquisition import (
    WeightedAcquisition,
)

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import AcquisitionFunction
    from torch import Tensor

    from neps.sampling.priors import Prior
    from neps.search_spaces.domain import Domain


def apply_pibo_acquisition_weight(
    acq_values: Tensor,
    X: Tensor,
    acq: AcquisitionFunction,
    *,
    prior: Prior,
    x_domain: Domain | list[Domain],
    prior_exponent: float,
):
    if acq._log:
        weighted_log_probs = prior.log_prob(X, frm=x_domain) + prior_exponent
        return acq_values + weighted_log_probs

    weighted_probs = prior.prob(X, frm=x_domain).pow(prior_exponent)
    return acq_values * weighted_probs


def pibo_acquisition(
    acq_fn: AcquisitionFunction,
    prior: Prior,
    prior_exponent: float,
    x_domain: Domain | list[Domain],
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
