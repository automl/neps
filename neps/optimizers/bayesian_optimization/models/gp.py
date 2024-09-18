"""Gaussian Process models for Bayesian Optimization."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeVar

import gpytorch
import gpytorch.constraints
import torch
from botorch.acquisition.analytic import SingleTaskGP
from botorch.models.gp_regression import (
    get_covar_module_with_dim_scaled_prior,
)
from botorch.models.gp_regression_mixed import CategoricalKernel, OutcomeTransform
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch.kernels import ScaleKernel
from torch._dynamo.utils import product

from neps.search_spaces.encoding import (
    CategoricalToIntegerTransformer,
    TensorEncoder,
    TensorPack,
)

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction

logger = logging.getLogger(__name__)


T = TypeVar("T")


def default_categorical_kernel(
    N: int,
    active_dims: tuple[int, ...] | None = None,
) -> ScaleKernel:
    """Default Categorical kernel for the GP."""
    # Following BoTorches implementation of the MixedSingleTaskGP
    return ScaleKernel(
        CategoricalKernel(
            ard_num_dims=N,
            active_dims=active_dims,
            lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-6),
        )
    )


def default_single_obj_gp(
    x: TensorPack,
    y: torch.Tensor,
    *,
    y_transform: OutcomeTransform | None = None,
) -> SingleTaskGP:
    """Default GP for single objective optimization."""
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    if y_transform is None:
        y_transform = Standardize(m=1)

    encoder = x.encoder
    numerics: list[int] = []
    categoricals: list[int] = []
    for hp_name, transformer in encoder.transformers.items():
        if isinstance(transformer, CategoricalToIntegerTransformer):
            categoricals.append(encoder.index_of[hp_name])
        else:
            numerics.append(encoder.index_of[hp_name])

    # Purely vectorial
    if len(categoricals) == 0:
        return SingleTaskGP(train_X=x.tensor, train_Y=y, outcome_transform=y_transform)

    # Purely categorical
    if len(numerics) == 0:
        return SingleTaskGP(
            train_X=x.tensor,
            train_Y=y,
            covar_module=default_categorical_kernel(len(categoricals)),
            outcome_transform=y_transform,
        )

    # Mixed
    numeric_kernel = get_covar_module_with_dim_scaled_prior(
        ard_num_dims=len(numerics),
        active_dims=tuple(numerics),
    )
    cat_kernel = default_categorical_kernel(
        len(categoricals), active_dims=tuple(categoricals)
    )

    # WARNING: I previously tried SingleTaskMixedGp which does the following:
    #
    # x K((x1, c1), (x2, c2)) =
    # x     K_cont_1(x1, x2) + K_cat_1(c1, c2) +
    # x      K_cont_2(x1, x2) * K_cat_2(c1, c2)
    #
    # In a toy example with a single binary categorical which acted like F * {0, 1},
    # the model collapsed to always predicting `0`. Causing all parameters defining F
    # to essentially be guess at random. This is a lot more stable but likely not as
    # good...
    # TODO: Figure out how to improve stability of this.
    kernel = numeric_kernel + cat_kernel

    return SingleTaskGP(
        train_X=x.tensor,
        train_Y=y,
        covar_module=kernel,
        outcome_transform=y_transform,
    )


def optimize_acq(
    acq_fn: AcquisitionFunction,
    encoder: TensorEncoder,
    *,
    n_candidates_required: int = 1,
    num_restarts: int = 20,
    n_intial_start_points: int | None = None,
    acq_options: Mapping[str, Any] | None = None,
    maximum_allowed_categorical_combinations: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize the acquisition function."""
    acq_options = acq_options or {}

    lower = [domain.lower for domain in encoder.domains]
    upper = [domain.upper for domain in encoder.domains]
    bounds = torch.tensor([lower, upper], dtype=torch.float64)

    cat_transformers = {
        name: t
        for name, t in encoder.transformers.items()
        if isinstance(t, CategoricalToIntegerTransformer)
    }
    if not any(cat_transformers):
        # Small heuristic to increase the number of candidates as our dimensionality
        # increases... we apply a cap.
        if n_intial_start_points is None:
            # TODO: Need to investigate how num_restarts is used in botorch to inform
            # this proxy.

            # Cap out at 4096 when len(bounds) >= 8
            n_intial_start_points = min(64 * len(bounds) ** 2, 4096)

        return optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,
            q=n_candidates_required,
            num_restarts=num_restarts,
            raw_samples=n_intial_start_points,
            **acq_options,
        )

    # We need to generate the product of all possible combinations of categoricals,
    # first we do a sanity check
    n_combos = reduce(
        lambda x, y: x * y, [len(t.choices) for t in cat_transformers.values()]
    )
    if n_combos > maximum_allowed_categorical_combinations:
        raise ValueError(
            "The number of fixed categorical dimensions is too high. "
            "This will lead to an explosion in the number of possible "
            f"combinations. Got: {n_combos} while the setting for the function"
            f" is: {maximum_allowed_categorical_combinations=}. Consider reducing the "
            "dimensions or consider encoding your categoricals in some other format."
        )

    # Right, now we generate all possible combinations
    # First, just collect the possible values per cat column
    # NOTE: Botorchs optim requires them to be as floats
    cats: dict[int, list[float]] = {
        encoder.index_of[name]: [float(i) for i in range(len(transformer.choices))]
        for name, transformer in cat_transformers.items()
    }

    # Second, generate all possible combinations
    fixed_cats: list[dict[int, float]]
    if len(cats) == 1:
        col, choice_indices = next(iter(cats.items()))
        fixed_cats = [{col: i} for i in choice_indices]
    else:
        fixed_cats = [
            dict(zip(cats.keys(), combo, strict=False))
            for combo in product(*cats.values())
        ]

    # TODO: we should deterministically shuffle the fixed_categoricals
    # as the underlying function does not.
    return optimize_acqf_mixed(
        acq_function=acq_fn,
        bounds=bounds,
        num_restarts=min(num_restarts // n_combos, 2),
        raw_samples=n_intial_start_points,
        q=n_candidates_required,
        fixed_features_list=fixed_cats,
        **acq_options,
    )
