from __future__ import annotations

import math

import torch


def ei(
    mu: torch.Tensor,
    cov: torch.Tensor,
    optimum: float | torch.Tensor,
    *,
    augmented_ei_regularizer: float | None = None,  # 0.01
    xi: float = 0.0,
    log_ei: bool = False,
    log_ei_epsilon: float = 1e-6,
) -> torch.Tensor:
    improvement = optimum - mu - xi

    sigma_sq = torch.diag(cov)
    sigma = torch.sqrt(sigma_sq)

    Z = improvement / sigma

    # If we calculate it ourselves, we save some computation as mu = 0
    # and sigma = 1 cancel a few terms out
    # https://en.wikipedia.org/wiki/Normal_distribution
    Z_cdf = 0.5 * (1 + torch.erf(Z / math.sqrt(2)))
    Z_pdf = 1 / (math.sqrt(2 * math.pi)) * torch.exp(-0.5 * Z**2)
    ei = improvement * Z_cdf + sigma * Z_pdf

    if augmented_ei_regularizer is not None:
        regularization_term = 1 + sigma_sq / augmented_ei_regularizer
        ei = ei / regularization_term

    if log_ei:
        ei = torch.log(ei + log_ei_epsilon)

    return ei


def acq_by_confidence(
    mu: torch.Tensor,
    cov: torch.Tensor,
    *,
    confidence_scale: float = 1.0,
) -> torch.Tensor:
    # Assumes we are trying to minimize our objective but
    # this acquisition function will be maximized, i.e. optimize
    # this function to find the point which is most likely to be
    # the minimum of the objective.

    #         ****
    #        * / \**
    #   ***** /   \- ****
    #  *     /      \    ***
    # *     /        \   |  *   ***
    #   ---/          \  |   +**
    # -/               \ | /   \
    #                   \|/     ---
    #                    -  <- lcb = mu - c * sigma
    # ______________________________
    lcb = mu - confidence_scale * torch.sqrt(torch.diag(cov))

    return -lcb  # Negate to make maximization


def weight_by_cost(
    acquisition_scores: torch.Tensor,
) -> torch.Tensor:
    # Assumes we are trying to minimize our objective but
    # this acquisition function will be maximized, i.e. optimize
    # this function to find the point which is most likely to be
    # the minimum of the objective.

    #         ****
    #        * / \**
    #   ***** /   \- ****
    #  *     /      \    ***
    # *     /        \   |  *   ***
    #   ---/          \  |   +**
    # -/               \ | /   \
    #                   \|/     ---
    #                    -  <- lcb = mu - c * sigma
    # ______________________________
    lcb = mu - cost_scale * torch.sqrt(torch.diag(cov))

    return -lcb  # Negate to make maximization
