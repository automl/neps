"""Gaussian Process models for Bayesian Optimization."""

from __future__ import annotations

import logging
import math
from functools import reduce
from typing import TYPE_CHECKING, Any, Mapping, TypeVar

import gpytorch
import gpytorch.constraints
import torch
from botorch.acquisition.analytic import SingleTaskGP
from botorch.models.gp_regression_mixed import (
    CategoricalKernel,
    Likelihood,
    OutcomeTransform,
)
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch.kernels import MaternKernel, ScaleKernel
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


def likelihood_with_prior_on_log_scale(
    mean: float = 1e-2,
    std: float = math.sqrt(3),
    bounds: tuple[float, float] = (1e-6, 1),
) -> gpytorch.likelihoods.GaussianLikelihood:
    """Default Gaussian likelihood with priors for the noise."""
    # The effect of the likelihood of noise is pretty crucial w.r.t.
    # whether we are going to overfit every point by overfitting with
    # the lengthscale, or whether we smooth through and assume variation
    # is due to noise. Setting it's prior is hard. For a non-noisy
    # function, we'd want it looooowww, like 1e-8 kind of low. For
    # even a 0.01% noise, we need that all the way up to 1e-2. Hence
    #
    # If we had 10% noise and we allow the noise to easily optimize towards
    # 1e-8, then the lengthscales are forced to become very small, essentially
    # overfitting. If we have 0% noise and we don't allow it to easily get low
    # then we will drastically underfit.
    # A guiding principle here is that we should allow the noise to be just
    # as if not slightly easier to tune than the lengthscales. I.e. we prefer
    # smoother functions as it is easier to acquisition over. However once we
    # over smooth and underfit, any new observations that inform us otherwise
    # could just be attributed to noise.
    #
    # TOOD: We may want to move the likelihood inside the GP and decay the
    # amount the GP can attribute to noise (reduce std and mean) relative
    # to samples seen, effectively reducing the smoothness of the GP overtime
    _noise_prior = gpytorch.priors.LogNormalPrior(math.log(mean) + std**2, std)
    return gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=_noise_prior,
        # Going below 1e-6 could introduuce a lot of numerical instability in the
        # kernels, even if it's a noiseless function
        noise_constraint=gpytorch.constraints.Interval(
            lower_bound=bounds[0],
            upper_bound=bounds[1],
            initial_value=mean,
        ),
    )


def default_signal_variance_prior() -> gpytorch.priors.NormalPrior:
    """Default prior for the signal variance."""
    # The outputscale prior is a bit more tricky. Essentially
    # it describes how much we expect the function to move
    # around the mean (0 as we normalize the `ys`)
    # Based on `Vanilla GP work great in High Dimensions` by Carl Hvafner
    # where it's fixed to `1.0`, we follow suit but allow some minor deviation
    # with a prior.
    return gpytorch.priors.NormalPrior(loc=1.0, scale=0.1)


def default_lengthscale_prior(
    N: int,
) -> tuple[gpytorch.priors.LogNormalPrior, gpytorch.constraints.Interval]:
    """Default prior for the lengthscale."""
    # Based on `Vanilla GP work great in High Dimensions` by Carl Hvafner
    # TODO: I'm not convinced entirely that the `std` is independant
    # of the dimension and number of samples
    lengthscale_prior = gpytorch.priors.LogNormalPrior(
        loc=math.sqrt(2.0) + math.log(N) / 2,
        scale=math.sqrt(3.0) * math.log(N),
    )
    # NOTE: It's possible to just specify `GreaterThan`, however
    # digging through the code, if this ends up at botorch's optimize,
    # it will read this and take the bounds and give it to Scipy's
    # L-BFGS-B optimizer. Without an upper bound, it defaults to `inf`,
    # which can impact gradient estimates.
    # tldr; set a bound if you have one, it always helps
    lengthscale_constraint = gpytorch.constraints.Interval(
        lower_bound=1e-4,
        upper_bound=1e3,
        initial_value=math.sqrt(2.0) + math.log(N) / 2,
    )
    return lengthscale_prior, lengthscale_constraint


def default_mean() -> gpytorch.means.ConstantMean:
    """Default mean for the GP."""
    return gpytorch.means.ConstantMean(
        constant_prior=gpytorch.priors.NormalPrior(0, 0.2),
        constant_constraint=gpytorch.constraints.Interval(
            lower_bound=-1e6,
            upper_bound=1e6,
            initial_value=0.0,
        ),
    )


def default_matern_kernel(
    N: int,
    active_dims: tuple[int, ...] | None = None,
) -> ScaleKernel:
    """Default Matern kernel for the GP."""
    lengthscale_prior, lengthscale_constraint = default_lengthscale_prior(N)

    return ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=N,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        ),
    )


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
    outcome_transform: OutcomeTransform | None = None,
) -> tuple[SingleTaskGP, Likelihood]:
    """Default GP for single objective optimization."""
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    if outcome_transform is None:
        outcome_transform = Standardize(m=1)

    encoder = x.encoder
    numerics: list[int] = []
    categoricals: list[int] = []
    for hp_name, transformer in encoder.transformers.items():
        if isinstance(transformer, CategoricalToIntegerTransformer):
            categoricals.append(encoder.index_of[hp_name])
        else:
            numerics.append(encoder.index_of[hp_name])

    # TODO: If we have a low cardinality integer, we should consider
    # just treating it as a categorical...
    likelihood = likelihood_with_prior_on_log_scale()

    # Purely vectorial
    if len(categoricals) == 0:
        gp = SingleTaskGP(
            train_X=x.tensor,
            train_Y=y,
            mean_module=default_mean(),
            likelihood=likelihood,
            # Only matern kernel
            covar_module=default_matern_kernel(len(numerics)),
            outcome_transform=outcome_transform,
        )
        return gp, likelihood

    # Purely categorical
    if len(numerics) == 0:
        gp = SingleTaskGP(
            train_X=x.tensor,
            train_Y=y,
            mean_module=default_mean(),
            likelihood=likelihood,
            # Only categorical kernel
            covar_module=default_categorical_kernel(len(categoricals)),
            outcome_transform=outcome_transform,
        )
        return gp, likelihood

    # Mixed
    numeric_kernel = default_matern_kernel(len(numerics), active_dims=tuple(numerics))
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
    # to essentially be guess at random. This is a lot more stable while testing...
    # TODO: Figure out why...
    kernel = numeric_kernel + cat_kernel

    gp = SingleTaskGP(
        train_X=x.tensor,
        train_Y=y,
        mean_module=default_mean(),
        likelihood=likelihood,
        covar_module=kernel,
        outcome_transform=outcome_transform,
    )
    return gp, likelihood


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
        fixed_cats = [dict(zip(cats.keys(), combo)) for combo in product(*cats.values())]

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
