"""Gaussian Process models for Bayesian Optimization."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import reduce
from itertools import product
from typing import TYPE_CHECKING, Any

import gpytorch.constraints
import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions.errors import OptimizationGradientError
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import Log, get_covar_module_with_dim_scaled_prior
from botorch.models.gp_regression_mixed import CategoricalKernel, OutcomeTransform
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Standardize
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_discrete_local_search,
    optimize_acqf_mixed,
)
from botorch.optim.initializers import initialize_q_batch
from botorch.optim.utils import fix_features
from botorch.utils.sampling import draw_sobol_samples
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel
from gpytorch.means import Mean as GpMean
from gpytorch.utils.warnings import NumericalWarning

from autoscan.exceptions import ConstraintViolationError
from autoscan.optimizers.acquisition import (
    WrappedAcquisition,
    cost_cooled_acq,
    pibo_acquisition,
)
from autoscan.space.encoding import CategoricalToIntegerTransformer, ConfigEncoder
from autoscan.utils.common import disable_warnings

if TYPE_CHECKING:
    from autoscan.sampling.priors import Prior
    from autoscan.space.parameters import Parameter
    from autoscan.state.trial import Trial

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=InputDataWarning)


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


class PowerLawMean(GpMean):
    """A fixed power law in the cost as the GP prior mean.

    `mean = slope * log(cost / reference_cost) + bias`, with the cost of an encoded
    configuration given by `cost_estimator`. Relative to `reference_cost`, the mean
    stays near `bias` instead of the large values the raw log cost would give.
    """

    def __init__(
        self,
        encoder: ConfigEncoder,
        cost_estimator: Callable[..., float],
        slope: float,
        bias: float,
        reference_cost: float = 1e16,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.cost_estimator = cost_estimator
        # Buffers, not parameters: the power law is fixed, not learned with the GP
        self.register_buffer("slope", torch.tensor(slope, dtype=torch.float64))
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float64))
        self.register_buffer(
            "reference_cost", torch.tensor(reference_cost, dtype=torch.float64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        with torch.no_grad():
            configs = self.encoder.decode(x.reshape(-1, x.shape[-1]))
            costs = torch.tensor(
                [self.cost_estimator(**config) for config in configs],
                dtype=x.dtype,
                device=x.device,
            )
        log_cost = torch.log((costs / self.reference_cost).clamp(min=1e-10))
        return (self.slope * log_cost + self.bias).view(batch_shape)


def make_default_single_obj_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    encoder: ConfigEncoder,
    *,
    y_transform: OutcomeTransform | None = None,
    cost_estimator: Callable[..., float] | None = None,
    min_cost: float = 1e16,
    mean_slope: float = -0.5,
    mean_bias: float = 5.0,
) -> SingleTaskGP:
    """Default GP for single objective optimization.

    With a `cost_estimator`, the prior mean is the power law `mean_slope *
    log(cost / min_cost) + mean_bias`, see `PowerLawMean`. It is not used for purely
    categorical spaces.
    """
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

    mean_module = (
        None
        if cost_estimator is None
        else PowerLawMean(
            encoder=encoder,
            cost_estimator=cost_estimator,
            slope=mean_slope,
            bias=mean_bias,
            reference_cost=min_cost,
        )
    )

    # Purely vectorial
    if len(categoricals) == 0:
        return SingleTaskGP(
            train_X=x, train_Y=y, outcome_transform=y_transform, mean_module=mean_module
        )

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
        train_X=x,
        train_Y=y,
        covar_module=kernel,
        outcome_transform=y_transform,
        mean_module=mean_module,
    )


def optimize_acq(  # noqa: C901, PLR0912, PLR0915
    acq_fn: AcquisitionFunction,
    encoder: ConfigEncoder,
    *,
    n_candidates_required: int = 1,
    num_restarts: int = 20,
    n_intial_start_points: int | None = None,
    acq_options: Mapping[str, Any] | None = None,
    fixed_features: dict[str, Any] | None = None,
    hide_warnings: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize the acquisition function.

    For purely numerical spaces, this uses botorch's `optimize_acqf()`.
    For purely categorical spaces, this uses `optimize_acqf_discrete_local_search()`.
    For mixed spaces, this uses a two step sequential optimization:
    1. Optimize acquisition over continuous space, sampling a random
        categorical combination to fix the continuous acquisition function
    2. Wrap the acquisition function to fix the numerical dimensions
        and optimize over the categorical space using
        `optimize_acqf_discrete_local_search()`
        NOTE: `optimize_acqf_discrete_local_search()` scales much better than
        `optimize_acqf_mixed()` for large categorical dimensions in the search space.


    Args:
        acq_fn: The acquisition function to optimize.
        encoder: The encoder used for encoding the configurations
        n_candidates_required: The number of candidates to return.
        num_restarts: The number of restarts to use during optimization.
        n_intial_start_points: The number of initial start points to use during
            optimization.
        acq_options: Additional options to pass to the botorch `optimizer_acqf` function.
        fixed_features: The features to fix to a certain value during acquisition.
        hide_warnings: Whether to hide numerical warnings issued during GP routines.

    Returns:
        The (encoded) optimized candidate(s) and corresponding acquisition value(s).
    """
    warning_context = (
        disable_warnings(NumericalWarning) if hide_warnings else nullcontext()
    )
    # Copy, as we add the `ic_generator` below and must not mutate the caller's dict
    acq_options = dict(acq_options or {})

    # BoTorch requires an `ic_generator` producing feasible initial conditions when
    # there are non-linear constraints. The discrete optimizers don't support them,
    # so for those we mask the acquisition value of infeasible candidates instead.
    nonlinear_constraints = acq_options.get("nonlinear_inequality_constraints")
    # Only accepted by `optimize_acqf()`, not by `optimize_acqf_mixed()`
    continuous_only_options: dict[str, Any] = {}
    if nonlinear_constraints is not None:
        acq_options.setdefault("ic_generator", make_ic_generator(nonlinear_constraints))
        # Black-box constraints are piecewise constant in integer/categorical dims,
        # so SLSQP mostly hits its iteration limit when the optimum is on the
        # boundary. Cap the iterations and don't retry, as BoTorch then falls back
        # to the feasible initial conditions anyway.
        acq_options["options"] = {
            "maxiter": _CONSTRAINED_SLSQP_MAXITER,
            **acq_options.get("options", {}),
        }
        continuous_only_options["retry_on_optimization_warning"] = False
        discrete_acq_fn: AcquisitionFunction = _FeasibilityMaskedAcquisition(
            acq_fn, nonlinear_constraints
        )
    else:
        discrete_acq_fn = acq_fn
    discrete_acq_options = {
        k: v
        for k, v in acq_options.items()
        if k not in ("nonlinear_inequality_constraints", "ic_generator", "options")
    }

    _fixed_features: dict[int, float] = {}
    if fixed_features is not None:
        for name, value in fixed_features.items():
            encoded_value = encoder.transformers[name].encode_one(value)
            encoded_index = encoder.index_of[name]
            _fixed_features[encoded_index] = encoded_value

    lower = [domain.lower for domain in encoder.domains]
    upper = [domain.upper for domain in encoder.domains]
    _acq_device = next(acq_fn.parameters(), torch.tensor(0.0)).device
    bounds = torch.tensor([lower, upper], dtype=torch.float64, device=_acq_device)

    cat_transformers = {
        name: t
        for name, t in encoder.transformers.items()
        if (
            name not in _fixed_features  # Don't include those which are fixed by caller
            and t.domain.is_categorical  # Only include categoricals
        )
    }

    num_numericals = len(encoder.domains) - len(cat_transformers)

    # Proceed with regular numerical acquisition
    if not any(cat_transformers):
        # Small heuristic to increase the number of candidates as our
        # dimensionality increases... we apply a cap of 4096,
        # which occurs when len(bounds) >= 8
        # TODO: Need to investigate how num_restarts is fully used in botorch to inform.
        if n_intial_start_points is None:
            n_intial_start_points = min(64 * len(bounds) ** 2, 4096)

        with warning_context:
            try:
                return optimize_acqf(  # type: ignore
                    acq_function=acq_fn,
                    bounds=bounds,
                    q=n_candidates_required,
                    num_restarts=num_restarts,
                    raw_samples=n_intial_start_points,
                    fixed_features=_fixed_features,
                    **acq_options,
                    **continuous_only_options,
                )
            except OptimizationGradientError as err:
                # TODO: Handle this better, maybe by retrying with more
                # initial points or restarts
                raise err

    # First, just collect the possible values per cat column
    # {hp_name: [v1, v2], hp_name2: [v1, v2, v3], ...}
    cats: dict[int, list[float]] = {
        encoder.index_of[name]: [
            float(i)  # NOTE: Botorchs optim requires them to be as floats
            for i in range(transformer.domain.cardinality)  # type: ignore
        ]
        for name, transformer in cat_transformers.items()
    }
    cat_keys = list(cats.keys())
    choices = [
        torch.tensor(cats[k], dtype=torch.float, device=_acq_device) for k in cat_keys
    ]
    fixed_cat: dict[int, float] = {}

    n_combos = reduce(
        lambda x, y: x * y,  # type: ignore
        [t.domain.cardinality for t in cat_transformers.values()],
        1,
    )

    if n_combos < num_restarts:
        # NOTE: Discrete local search only seems to work when the number of categorical
        # combinations is more than num_restarts, else it's unable to generate enough
        # candidates. Since num_restarts is usually set pretty low, we can just use
        # optimize_acqf_mixed in this case, without incurring too high of a computational
        # cost.

        # Generate all possible categorical combinations and fix them during optimization
        # for the outer loop inside optimize_acqf_mixed
        fixed_cats: list[dict[int, float]]
        if len(cats) == 1:
            col, choice_indices = next(iter(cats.items()))
            fixed_cats = [{col: i} for i in choice_indices]
        else:
            fixed_cats = [
                dict(zip(cats.keys(), combo, strict=False))
                for combo in product(*cats.values())
            ]
        with warning_context:
            return optimize_acqf_mixed(  # type: ignore
                acq_function=acq_fn,
                bounds=bounds,
                num_restarts=min(num_restarts // n_combos, 2),
                raw_samples=n_intial_start_points,
                q=n_candidates_required,
                fixed_features_list=fixed_cats,
                **acq_options,
            )

    if num_numericals > 0:
        with warning_context:
            # Sample a random categorical combination and keep it fixed during
            # the continuous optimization step
            fixed_cat = {key: float(np.random.choice(cats[key])) for key in cat_keys}

            # Step 1: Optimize acquisition function over the continuous space

            if len(_fixed_features) > 0:
                fixed_cat.update(_fixed_features)

            best_x_continuous, _ = optimize_acqf(
                acq_function=acq_fn,
                bounds=bounds,
                q=n_candidates_required,
                num_restarts=num_restarts,
                raw_samples=n_intial_start_points,
                fixed_features=fixed_cat,
                **acq_options,
                **continuous_only_options,
            )

            # Extract the numerical dims from the optimized continuous vector
            fixed_numericals = {
                i: float(best_x_continuous[0, i].item())
                for i in range(len(encoder.domains))
                if i not in cat_keys
            }

            # Update fixed_numericals with _fixed_features
            fixed_numericals.update(_fixed_features)

            # Step 2: Wrap acquisition function for discrete search
            wrapped_acq = WrappedAcquisition(
                acq=discrete_acq_fn,
                encoder=encoder,
                fixed_numericals=fixed_numericals,
            )

            # Step 3: Run discrete local search over the categorical space
            # with the wrapped acquisition function
            best_cat_tensor, _ = optimize_acqf_discrete_local_search(
                acq_function=wrapped_acq,
                discrete_choices=choices,
                q=n_candidates_required,
                num_restarts=num_restarts,
                raw_samples=n_intial_start_points,
            )

            # Step 4: Concatenate best categorical and numerical dims, along with
            # any fixed features provided by the caller

            q, c_dims = best_cat_tensor.shape
            n_dims = len(fixed_numericals)
            new_X_shape = (q, c_dims + n_dims)

            # Create a new tensor to hold the concatenated dimensions
            best_x_full: torch.Tensor = torch.empty(
                new_X_shape, dtype=best_cat_tensor.dtype, device=best_cat_tensor.device
            )

            # Create a mask to identify positions of categorical and numerical dimensions
            mask = torch.ones(
                c_dims + n_dims, dtype=torch.bool, device=best_cat_tensor.device
            )
            insert_idxs = torch.tensor(
                list(fixed_numericals.keys()), device=best_cat_tensor.device
            )
            mask[insert_idxs] = False

            # Fill in the fixed numerical values and the input categorical values
            for idx, val in fixed_numericals.items():
                best_x_full[:, idx] = val
            best_x_full[:, mask] = best_cat_tensor

            # Evaluate the final acquisition value
            # The categorical search may not find a feasible combination for the
            # fixed numericals, fall back to the feasible continuous candidates then
            if nonlinear_constraints is not None:
                infeasible = ~_feasibility_mask(best_x_full, nonlinear_constraints)
                best_x_full[infeasible] = best_x_continuous[infeasible]

            with torch.no_grad():
                best_val_final = acq_fn(best_x_full)

            return best_x_full, best_val_final

    else:
        with warning_context:
            return optimize_acqf_discrete_local_search(  # type: ignore
                acq_function=discrete_acq_fn,
                discrete_choices=choices,
                q=n_candidates_required,
                num_restarts=num_restarts,
                raw_samples=n_intial_start_points,
                **discrete_acq_options,
            )


def encode_trials_for_gp(
    trials: Mapping[str, Trial],
    parameters: Mapping[str, Parameter],
    *,
    encoder: ConfigEncoder | None = None,
    device: torch.device | None = None,
) -> tuple[GPEncodedData, ConfigEncoder]:
    """Encode the trials for use in a GP.

    Args:
        trials: The trials to encode.
        encoder: The encoder to use. If `None`, one will be created.
        device: The device to use.

    Returns:
        The encoded data and the encoder
    """
    train_configs: list[Mapping[str, Any]] = []
    train_losses: list[float] | list[Sequence[float]] = []
    train_costs: list[float] = []
    pending_configs: list[Mapping[str, Any]] = []

    if encoder is None:
        encoder = ConfigEncoder.from_parameters(parameters)

    for trial in trials.values():
        if trial.report is None:
            pending_configs.append(trial.config)
            continue

        train_configs.append(trial.config)

        objective_to_minimize = trial.report.objective_to_minimize
        train_losses.append(
            torch.nan if objective_to_minimize is None else objective_to_minimize
        )

        cost = trial.report.cost
        train_costs.append(torch.nan if cost is None else cost)

    x_train = encoder.encode(train_configs, device=device)
    y_train = torch.tensor(train_losses, dtype=torch.float64, device=device)

    # OPTIM: The issue here is that the error could be a bug, in which case
    # if the user restarts, we don't want to too heavily penalize that area.
    # On the flip side, if the configuration is actually what's causing the
    # crashes, then we just want to ensure that the GP is discouraged from
    # visiting that area. Setting to the median also ensures that the GP does
    # not end up with a highly skewed function apprxoimation, for example,
    # setting tiny lengthscales, to ensure it can model the sharp change
    # in the performance around the crashed config.
    fill_value = torch.nanmedian(y_train).item()
    y_train = torch.nan_to_num(y_train, nan=fill_value)

    cost_train = torch.tensor(train_costs, dtype=torch.float64, device=device)
    if len(pending_configs) > 0:
        x_pending = encoder.encode(pending_configs, device=device)
    else:
        x_pending = None

    data = GPEncodedData(x=x_train, y=y_train, cost=cost_train, x_pending=x_pending)
    return data, encoder


def fit_and_acquire_from_gp(
    *,
    gp: SingleTaskGP,  # noqa: ARG001
    x_train: torch.Tensor,
    encoder: ConfigEncoder,
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
    n_initial_start_points: int | None = 256,
    fixed_acq_features: dict[str, Any] | None = None,
    acq_options: Mapping[str, Any] | None = None,
    hide_warnings: bool = False,
) -> torch.Tensor:
    """Acquire the next configuration to evaluate using a GP.

    Please see the following for:

    * Making a GP to pass in:
        [`make_default_single_obj_gp()`][autoscan.optimizers.models.gp.make_default_single_obj_gp]
    * Encoding configurations:
        [`encode_trials_for_gp()`][autoscan.optimizers.models.gp.encode_trials_for_gp]

    Args:
        gp: The GP model to use.
        x_train: The encoded configurations that have already been evaluated
        encoder: The encoder used for encoding the configurations
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
        cost_percentage_used: The percentage of the budget used so far. This is used to
            determine the strength of the cost cooling. Should be between 0 and 1.
            Must be provided if costs is provided.
        costs_on_log_scale: Whether the costs are on a log scale.
        encoder: The encoder used for encoding the configurations
        seed: The seed to use.
        fixed_acq_features: The features to fix to a certain value during acquisition.
        n_candidates_required: The number of candidates to return. If left
            as `None`, only the best candidate will be returned. Otherwise
            a list of candidates will be returned.
        num_restarts: The number of restarts to use during optimization.
        n_initial_start_points: The number of initial start points to use during
            optimization. If `None`, a dimension-dependent heuristic is used
            (see [`optimize_acq()`][autoscan.optimizers.models.gp.optimize_acq]).
        acq_options: Additional options to pass to the botorch `optimizer_acqf` function.
        hide_warnings: Whether to hide numerical warnings issued during GP routines.

    Returns:
        The encoded next configuration(s) to evaluate. Use the encoder you provided
        to decode the configuration.
    """
    if seed is not None:
        raise NotImplementedError("Seed is not implemented yet for gps")

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
                "Must have at least some configurations reported with a cost"
                " if using costs with a GP."
            )

        if missing_costs.any():
            not_missing_mask = ~missing_costs
            x_train_cost = x_train[not_missing_mask]
            y_train_cost = costs[not_missing_mask]
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
            used_max_cost_total_percentage=cost_percentage_used,
        )

    _n = n_candidates_required if n_candidates_required is not None else 1

    candidates, _scores = optimize_acq(
        acquisition,
        encoder,
        n_candidates_required=_n,
        num_restarts=num_restarts,
        n_intial_start_points=n_initial_start_points,
        fixed_features=fixed_acq_features,
        acq_options=acq_options,
        hide_warnings=hide_warnings,
    )
    return candidates


INFEASIBLE_CONSTRAINT_VALUE = -1e6
"""Constraint value used for configurations whose constraint can not be evaluated."""

_INFEASIBLE_ACQ_VALUE = -1e10
"""Acquisition value given to infeasible candidates in the discrete optimizers."""

_FD_MIN_STEP = 1e-3
_FD_MAX_STEP = 0.5
_IC_MAX_ATTEMPTS = 4
_CONSTRAINED_SLSQP_MAXITER = 50


def _evaluate_constraint(
    constraint_func: Callable[[Mapping[str, Any]], float],
    config: Mapping[str, Any],
) -> tuple[float, bool]:
    """Evaluate a black-box constraint, mapping failures to an infeasible value.

    Returns:
        The constraint value and whether the evaluation succeeded.
    """
    try:
        val = float(constraint_func(config))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Constraint evaluation failed for config {config}: {e}")
        return INFEASIBLE_CONSTRAINT_VALUE, False

    if not np.isfinite(val):
        logger.debug(f"Constraint returned non-finite value {val} for config {config}")
        return INFEASIBLE_CONSTRAINT_VALUE, False

    return val, True


def _finite_difference_step(
    x: torch.Tensor,
    dim: int,
    base_config: Mapping[str, Any],
    encoder: ConfigEncoder,
) -> tuple[Mapping[str, Any], float] | None:
    """Find the smallest step along `dim` that changes the decoded configuration.

    Integer and categorical dimensions are rounded on decode, so a fixed tiny step
    mostly decodes to the same configuration. We therefore grow the step until the
    decoded configuration changes, stepping backwards near the upper bound so that
    we never leave the domain.

    Returns:
        The perturbed configuration and the signed step, or `None` if no step within
        `_FD_MAX_STEP` changes the configuration.
    """
    domain = encoder.domains[dim]
    width = domain.upper - domain.lower
    h = _FD_MIN_STEP * width
    while h <= _FD_MAX_STEP * width:
        if x[dim] + h <= domain.upper:
            signed_h = h
        elif x[dim] - h >= domain.lower:
            signed_h = -h
        else:
            return None

        x_p = x.clone()
        x_p[dim] += signed_h
        config_p = encoder.decode(x_p.unsqueeze(0).cpu())[0]
        if config_p != base_config:
            return config_p, signed_h
        h *= 2
    return None


class BlackBoxConstraintFn(torch.autograd.Function):
    """Wraps a black-box constraint on configurations as a differentiable function
    of the encoded tensor, using finite differences for the gradient.

    The constraint follows the BoTorch convention: `>= 0` is feasible. Configurations
    for which the constraint fails or is non-finite are treated as infeasible.
    """

    @staticmethod
    def forward(ctx, x, constraint_func, encoder):  # type: ignore
        x_flat = x.detach().reshape(-1, x.shape[-1])
        configs = encoder.decode(x_flat.cpu())
        results = [_evaluate_constraint(constraint_func, c) for c in configs]

        n_failed = sum(not ok for _, ok in results)
        if n_failed > 0:
            logger.warning(
                f"Constraint evaluation failed for {n_failed}/{len(configs)} "
                "configurations, treating them as infeasible."
            )

        ctx.constraint_func = constraint_func
        ctx.encoder = encoder
        ctx.configs = configs
        ctx.results = results
        ctx.save_for_backward(x)

        vals = [val for val, _ in results]
        # (..., d) -> (...), e.g. (d,) -> () and (b, q, d) -> (b, q)
        return torch.tensor(vals, dtype=x.dtype, device=x.device).view(x.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        (x,) = ctx.saved_tensors
        x_flat = x.detach().reshape(-1, x.shape[-1])
        grad_out_flat = grad_output.reshape(-1)
        grad_in_flat = torch.zeros_like(x_flat)

        for i in range(len(x_flat)):
            base_val, base_ok = ctx.results[i]
            # Skip points that don't contribute, or where we have no base value
            if grad_out_flat[i] == 0 or not base_ok:
                continue

            for d in range(x_flat.shape[1]):
                step = _finite_difference_step(x_flat[i], d, ctx.configs[i], ctx.encoder)
                if step is None:
                    continue

                config_p, h = step
                val_p, ok = _evaluate_constraint(ctx.constraint_func, config_p)
                if not ok:
                    continue

                slope = float(np.clip((val_p - base_val) / h, -1e6, 1e6))
                grad_in_flat[i, d] = slope * grad_out_flat[i]

        return grad_in_flat.view_as(x), None, None


def encode_constraints_func(
    constraints_func: Callable[[Mapping[str, Any]], float],
    *,
    encoder: ConfigEncoder,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Turn a constraint on configurations into a constraint on encoded tensors."""

    def inner(x: torch.Tensor) -> torch.Tensor:
        return BlackBoxConstraintFn.apply(x, constraints_func, encoder)

    return inner


def _feasibility_mask(
    X: torch.Tensor,
    constraints: Sequence[tuple[Callable[[torch.Tensor], torch.Tensor], bool]],
) -> torch.Tensor:
    """Return a mask of shape `X.shape[:-1]` of points satisfying all constraints."""
    mask = torch.ones(X.shape[:-1], dtype=torch.bool, device=X.device)
    with torch.no_grad():
        for constraint, _ in constraints:
            mask &= constraint(X) >= 0
    return mask


def make_ic_generator(
    constraints: Sequence[tuple[Callable[[torch.Tensor], torch.Tensor], bool]],
) -> Callable[..., torch.Tensor]:
    """Create an initial condition generator that respects non-linear constraints.

    Feasible points are collected into a pool first and q-batches are assembled
    from it, so the share of feasible starting points does not decay with `q`.
    """

    def ic_generator(
        acq_function: AcquisitionFunction,
        bounds: torch.Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int,
        fixed_features: dict[int, float] | None = None,
        options: Mapping[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> torch.Tensor:
        options = options or {}
        seed = options.get("seed")
        if seed is None:
            # Draw from the global RNG so `torch.manual_seed` makes this reproducible
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        n_needed = num_restarts * q
        n_draw = raw_samples * q
        n_drawn = 0
        pools: list[torch.Tensor] = []
        for attempt in range(_IC_MAX_ATTEMPTS):
            X = draw_sobol_samples(bounds=bounds, n=n_draw, q=1, seed=seed + attempt)
            X = fix_features(X.squeeze(-2), fixed_features)
            n_drawn += len(X)
            pools.append(X[_feasibility_mask(X, constraints)])
            if sum(len(p) for p in pools) >= n_needed:
                break
            n_draw *= 4

        pool = torch.unique(torch.cat(pools), dim=0)
        if len(pool) == 0:
            raise ConstraintViolationError(
                f"No feasible initial conditions found in {n_drawn} samples."
            )
        if len(pool) < n_needed:
            logger.warning(
                f"Constraint is very strict: only {len(pool)} feasible points found "
                f"in {n_drawn} samples for {num_restarts} restarts of q={q}."
            )

        # Assemble q-batches from the feasible pool
        generator = torch.Generator().manual_seed(seed)
        n_batches = max(raw_samples, num_restarts)
        if len(pool) >= n_batches * q:
            idx = torch.randperm(len(pool), generator=generator)[: n_batches * q]
            idx = idx.view(n_batches, q)
        else:
            idx = torch.randint(len(pool), (n_batches, q), generator=generator)
            idx = torch.unique(idx, dim=0)
        X_batches = pool[idx.to(pool.device)]

        with torch.no_grad():
            acq_vals = acq_function(X_batches)

        finite = torch.isfinite(acq_vals)
        if not finite.any():
            raise RuntimeError(
                "All initial conditions produced non-finite acquisition values."
            )
        X_batches, acq_vals = X_batches[finite], acq_vals[finite]

        if len(X_batches) <= num_restarts:
            return X_batches
        return initialize_q_batch(X_batches, acq_vals, num_restarts)

    return ic_generator


class _FeasibilityMaskedAcquisition(AcquisitionFunction):
    """Assigns a very low acquisition value to infeasible candidates.

    Used for the discrete optimizers, which do not support non-linear constraints.
    """

    def __init__(
        self,
        acq: AcquisitionFunction,
        constraints: Sequence[tuple[Callable[[torch.Tensor], torch.Tensor], bool]],
    ) -> None:
        super().__init__(model=acq.model)
        self.acq = acq
        self.constraints = constraints

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        vals = self.acq(X)
        # A q-batch is only feasible if all of its points are
        feasible = _feasibility_mask(X, self.constraints).all(dim=-1)
        return torch.where(feasible, vals, torch.full_like(vals, _INFEASIBLE_ACQ_VALUE))
