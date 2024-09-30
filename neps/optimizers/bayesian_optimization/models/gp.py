"""Gaussian Process models for Bayesian Optimization."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeVar
from dataclasses import dataclass

from botorch.fit import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood
import torch
import gpytorch.constraints
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import Log, get_covar_module_with_dim_scaled_prior
from botorch.models.gp_regression_mixed import CategoricalKernel, OutcomeTransform
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Standardize
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch.kernels import ScaleKernel
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from itertools import product

from neps.optimizers.bayesian_optimization.acquisition_functions.cost_cooling import (
    cost_cooled_acq,
)
from neps.optimizers.bayesian_optimization.acquisition_functions.pibo import (
    pibo_acquisition,
)
from neps.sampling.priors import Prior
from neps.search_spaces.encoding import CategoricalToIntegerTransformer, ConfigEncoder
from neps.search_spaces.search_space import SearchSpace
from neps.state.trial import Trial

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction

logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass
class GPEncodedData:
    """Tensor data of finished configurations."""

    x: torch.Tensor
    y: torch.Tensor
    cost: torch.Tensor | None = None
    x_pending: torch.Tensor | None = None


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


def make_default_single_obj_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    encoder: ConfigEncoder,
    *,
    y_transform: OutcomeTransform | None = None,
) -> SingleTaskGP:
    """Default GP for single objective optimization."""
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    if y_transform is None:
        y_transform = Standardize(m=1)

    numerics: list[int] = []
    categoricals: list[int] = []
    for hp_name, transformer in encoder.transformers.items():
        if isinstance(transformer, CategoricalToIntegerTransformer):
            categoricals.append(encoder.index_of[hp_name])
        else:
            numerics.append(encoder.index_of[hp_name])

    # Purely vectorial
    if len(categoricals) == 0:
        return SingleTaskGP(train_X=x, train_Y=y, outcome_transform=y_transform)

    # Purely categorical
    if len(numerics) == 0:
        return SingleTaskGP(
            train_X=x,
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
        train_X=x, train_Y=y, covar_module=kernel, outcome_transform=y_transform
    )


def optimize_acq(
    acq_fn: AcquisitionFunction,
    encoder: ConfigEncoder,
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


def encode_trials_for_gp(
    trials: Mapping[str, Trial],
    space: SearchSpace,
    *,
    encoder: ConfigEncoder | None = None,
    device: torch.device | None = None,
) -> tuple[GPEncodedData, ConfigEncoder]:
    train_configs: list[Mapping[str, Any]] = []
    train_losses: list[float] = []
    train_costs: list[float] = []
    pending_configs: list[Mapping[str, Any]] = []

    if encoder is None:
        encoder = ConfigEncoder.default({**space.numerical, **space.categoricals})

    for trial in trials.values():
        if trial.report is None:
            pending_configs.append(trial.config)
            continue

        train_configs.append(trial.config)

        loss = trial.report.loss
        train_losses.append(torch.nan if loss is None else loss)

        cost = trial.report.cost
        train_costs.append(torch.nan if cost is None else cost)

    x_train = encoder.encode(train_configs, device=device)
    y_train = torch.tensor(train_losses, dtype=torch.float64, device=device)
    cost_train = torch.tensor(train_costs, dtype=torch.float64, device=device)
    if len(pending_configs) > 0:
        x_pending = encoder.encode(pending_configs, device=device)
    else:
        x_pending = None

    data = GPEncodedData(x=x_train, y=y_train, cost=cost_train, x_pending=x_pending)
    return data, encoder


def fit_and_acquire_from_gp(
    *,
    gp: SingleTaskGP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    encoder: ConfigEncoder,
    fantasize_pending: torch.Tensor | None = None,
    acquisition: AcquisitionFunction,
    prior: Prior | None = None,
    pibo_exp_term: float | None = None,
    cost_gp: SingleTaskGP | None = None,
    costs: torch.Tensor | None = None,
    cost_percentage_used: float | None = None,
    costs_on_log_scale: bool = True,
    seed: int | None = None,
    n_candidates_required: int | None = None,
    num_restarts: int = 20,
    n_initial_start_points: int | None = None,
    maximum_allowed_categorical_combinations: int = 30,
    acq_options: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Acquire the next configuration to evaluate using a GP.

    Please see the following for:

    * Making a GP to pass in:
        [`make_default_single_obj_gp`][neps.optimizers.bayesian_optimization.models.gp.make_default_single_obj_gp]
    * Encoding configurations:
        [`encode_trails_for_gp`][neps.optimizers.bayesian_optimization.models.gp.encode_trails_for_gp]

    Args:
        gp: The GP model to use.
        x_train: The encoded configurations that have already been evaluated
        y_train: The loss of the evaluated configurations.

            !!! note "NaNs"

                Any y values encoded with NaNs will automatically be filled with
                the mean loss value. This is to ensure a smoother acquisition
                function optimization landscape. While this is a poorer
                approximation of the landscape, this does not matter as we are
                aiming to ensure that the GP models good areas as being better
                than any other area, regardless of whether they are average or garbage.

        encoder: The encoder used for encoding the configurations
        fantasize_pending: The pending configurations to fantasize over. Please be aware
            that there are more efficient strategies as some acquisition functions can
            handle this explicitly.
        acquisition: The acquisition function to use.

            A good default is `qLogNoisyExpectedImprovement` which can
            handle pending configurations gracefully without fantasization.

        prior: The prior to use over configurations. If this is provided, the
            acquisition function will be further weighted using the piBO acquisition.
        pibo_exp_term: The exponential term for the piBO acquisition. If `None` is
            provided, one will be estimated.
        costs: The costs of evaluating the configurations. If this is provided,
            then a secondary GP will be used to estimate the cost of a given
            configuration and factor into the weighting during the acquisiton of a new
            configuration.
        cost_percentage_used: The percentage of the budget used so far. This is used to determine
            the strength of the cost cooling. Should be between 0 and 1.
            Must be provided if costs is provided.
        costs_on_log_scale: Whether the costs are on a log scale.
        encoder: The encoder used for encoding the configurations
        seed: The seed to use.
        n_candidates_required: The number of candidates to return. If left
            as `None`, only the best candidate will be returned. Otherwise
            a list of candidates will be returned.
        num_restarts: The number of restarts to use during optimization.
        n_initial_start_points: The number of initial start points to use during optimization.
        maximum_allowed_categorical_combinations: The maximum number of categorical
            combinations to allow. If the number of combinations exceeds this, an error
            will be raised.
        acq_options: Additional options to pass to the botorch `optimizer_acqf` function.

    Returns:
        The encoded next configuration(s) to evaluate. Use the encoder you provided
        to decode the configuration.
    """
    fit_gpytorch_mll(ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp))

    if fantasize_pending is not None:
        y_train = torch.cat([y_train, gp.posterior(fantasize_pending).mean], dim=0)
        x_train = torch.cat([x_train, fantasize_pending], dim=0)

    if prior:
        if pibo_exp_term is None:
            raise ValueError(
                "If providing a prior, you must provide the `pibo_exp_term`."
            )

        acquisition = pibo_acquisition(
            acquisition,
            prior=prior,
            prior_exponent=pibo_exp_term,
            x_domain=encoder.domains,
        )

    if costs is not None:
        if cost_percentage_used is None:
            raise ValueError(
                "If providing costs, you must provide `cost_percentage_used`."
            )

        # We simply ignore missing costs when training the cost GP.
        missing_costs = torch.isnan(costs)
        if missing_costs.any():
            raise ValueError(
                "Must have at least some configurations reported with a cost if using costs"
                " with a GP."
            )

        if missing_costs.any():
            not_missing_mask = ~missing_costs
            x_train_cost = costs[not_missing_mask]
            y_train_cost = x_train[not_missing_mask]
        else:
            x_train_cost = x_train
            y_train_cost = costs

        if costs_on_log_scale:
            transform = ChainedOutcomeTransform(
                log=Log(),
                standardize=Standardize(m=1),
            )
        else:
            transform = Standardize(m=1)

        cost_gp = make_default_single_obj_gp(
            x_train_cost,
            y_train_cost,
            encoder=encoder,
            y_transform=transform,
        )
        fit_gpytorch_mll(
            ExactMarginalLogLikelihood(likelihood=cost_gp.likelihood, model=cost_gp)
        )
        acquisition = cost_cooled_acq(
            acq_fn=acquisition,
            model=cost_gp,
            used_budget_percentage=cost_percentage_used,
        )

    _n = n_candidates_required if n_candidates_required is not None else 1

    candidates, _scores = optimize_acq(
        acquisition,
        encoder,
        n_candidates_required=_n,
        num_restarts=num_restarts,
        n_intial_start_points=n_initial_start_points,
        acq_options=acq_options,
        maximum_allowed_categorical_combinations=maximum_allowed_categorical_combinations,
    )
    return candidates
