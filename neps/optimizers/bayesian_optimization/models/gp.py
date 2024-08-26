from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Mapping, TypeVar

import gpytorch
import gpytorch.constraints
import torch
from botorch.acquisition.analytic import SingleTaskGP
from botorch.models import MixedSingleTaskGP
from botorch.models.gp_regression_mixed import CategoricalKernel
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch.kernels import MaternKernel, ScaleKernel

from neps.search_spaces.encoding import (
    CategoricalToIntegerTransformer,
    DataEncoder,
    DataPack,
)

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction

logger = logging.getLogger(__name__)


T = TypeVar("T")


def default_likelihood_with_prior() -> gpytorch.likelihoods.GaussianLikelihood:
    # The effect of the likelihood of noise is pretty crucial w.r.t.
    # whether we are going to overfit every point by overfitting with
    # the lengthscale, or whether we smooth through and assume variation
    # is due to noise. Setting it's prior is hard. For a non-noisy
    # function, we'd want it looooowww, like 1e-8 kind of low. For
    # even a 0.01% noise, we need that all the way up to 1e-2. Hence
    #
    # If we had 10% noise and we allow the noise to easily optimize towards
    # 1e-8, then the lengthscales are forced to beome very small, essentially
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
    noise_mean = 1e-2
    noise_std = math.sqrt(3)
    _noise_prior = gpytorch.priors.LogNormalPrior(
        math.log(noise_mean) + noise_std**2,
        noise_std,
    )
    return gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=_noise_prior,
        # Going below 1e-6 could introduuce a lot of numerical instability in the
        # kernels, even if it's a noiseless function
        noise_constraint=gpytorch.constraints.Interval(
            lower_bound=1e-6,
            upper_bound=1,
            initial_value=noise_mean,
        ),
    )


def default_signal_variance_prior() -> gpytorch.priors.NormalPrior:
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
    # Based on `Vanilla GP work great in High Dimensions` by Carl Hvafner
    # TODO: I'm not convinced entirely that the `std` is independant
    # of the dimension and number of samples
    lengthscale_prior = gpytorch.priors.LogNormalPrior(
        loc=math.sqrt(2.0) + math.log(N) / 2,
        scale=math.sqrt(3.0),
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
    return gpytorch.means.ConstantMean(
        constant_prior=gpytorch.priors.NormalPrior(0, 0.2),
        constant_constraint=gpytorch.constraints.Interval(
            lower_bound=-1e6,
            upper_bound=1e6,
            initial_value=0.0,
        ),
    )


def default_matern_kernel(
    N: int,  # noqa: N803
    active_dims: tuple[int, ...] | None = None,
) -> ScaleKernel:
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
    N: int,  # noqa: N803
    active_dims: tuple[int, ...] | None = None,
) -> ScaleKernel:
    # Following BoTorches implementation of the MixedSingleTaskGP
    return ScaleKernel(
        CategoricalKernel(
            ard_num_dims=N,
            active_dims=active_dims,
            lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-6),
        )
    )


def default_single_obj_gp(x: DataPack, y: torch.Tensor) -> SingleTaskGP:
    encoder = x.encoder
    assert x.tensor is not None
    assert encoder.tensors is not None
    # Here, we will collect all graph encoded hyperparameters and assign each
    # to its own individual WL kernel.
    if encoder.graphs is not None:
        raise NotImplementedError("Graphs are not yet supported.")

    numerics: list[str] = []
    categoricals: list[str] = []
    for hp_name, transformer in encoder.tensors.transformers.items():
        if isinstance(transformer, CategoricalToIntegerTransformer):
            categoricals.append(hp_name)
        else:
            numerics.append(hp_name)

    categorical_indices = encoder.indices(categoricals)
    numeric_indices = encoder.indices(numerics)

    # Purely vectorial
    if len(categorical_indices) == 0:
        return SingleTaskGP(
            train_X=x.tensor,
            train_Y=y,
            mean_module=default_mean(),
            likelihood=default_likelihood_with_prior(),
            # Only matern kernel
            covar_module=default_matern_kernel(len(numerics)),
            outcome_transform=Standardize(m=1),
        )

    # Purely categorical
    if len(numeric_indices) == 0:
        return SingleTaskGP(
            train_X=x.tensor,
            train_Y=y,
            mean_module=default_mean(),
            likelihood=default_likelihood_with_prior(),
            # Only categorical kernel
            covar_module=default_categorical_kernel(len(categoricals)),
            outcome_transform=Standardize(m=1),
        )

    # Mixed
    def cont_kernel_factory(
        batch_shape: torch.Size,
        ard_num_dims: int,
        active_dims: list[int],
    ) -> ScaleKernel:
        lengthscale_prior, lengthscale_constraint = default_lengthscale_prior(
            ard_num_dims
        )
        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                batch_shape=batch_shape,
                ard_num_dims=ard_num_dims,
                active_dims=active_dims,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint,
            ),
        )

    return MixedSingleTaskGP(
        train_X=x.tensor,
        train_Y=y,
        cat_dims=list(categorical_indices),
        likelihood=default_likelihood_with_prior(),
        cont_kernel_factory=cont_kernel_factory,
        outcome_transform=Standardize(m=1),
    )


def optimize_acq(
    acq_fn: AcquisitionFunction,
    encoder: DataEncoder,
    *,
    q: int,
    num_restarts: int,
    raw_samples: int,
    acq_options: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    acq_options = acq_options or {}
    if encoder.has_graphs():
        raise NotImplementedError("Graphs are not yet supported.")

    assert encoder.tensors is not None
    lower = [t.domain.lower for t in encoder.tensors.transformers.values()]
    upper = [t.domain.upper for t in encoder.tensors.transformers.values()]
    bounds = torch.tensor([lower, upper], dtype=torch.float)

    fixed_categoricals = encoder.categorical_product_indices()

    if not any(fixed_categoricals):
        return optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            **acq_options,
        )

    if len(fixed_categoricals) > 30:
        raise ValueError(
            "The number of fixed categorical dimensions is too high. "
            "This will lead to an explosion in the number of possible "
            "combinations. Please reduce the number of fixed categorical "
            "dimensions or consider encoding your categoricals in some other format."
        )

    # TODO: we should deterministicall shuffle the fixed_categoricals as the
    # underlying function does not.
    return optimize_acqf_mixed(
        acq_function=acq_fn,
        bounds=bounds,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        q=q,
        fixed_features_list=fixed_categoricals,  # type: ignore
        **acq_options,
    )
