"""Gaussian Process models for Bayesian Optimization."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence, Callable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import reduce
from itertools import product
from typing import TYPE_CHECKING, Any

import gpytorch.constraints
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import Log, get_covar_module_with_dim_scaled_prior
from botorch.models.gp_regression_mixed import CategoricalKernel, OutcomeTransform
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Standardize
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel
from gpytorch.utils.warnings import NumericalWarning
from gpytorch.means import Mean as GpMean
from neps.optimizers.acquisition import cost_cooled_acq, pibo_acquisition
from neps.space.encoding import CategoricalToIntegerTransformer, ConfigEncoder
from neps.utils.common import disable_warnings

from neps.sampling.samplers import Sobol
if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction

    from neps.sampling.priors import Prior
    from neps.space.parameters import Parameter
    from neps.state.trial import Trial

logger = logging.getLogger(__name__)


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

class ScalingMeanModule(gpytorch.means.Mean):
    """
    Learns a scaling law trend: y = sum(w * x^p) + bias.
    """
    def __init__(
        self, 
        scaling_dims: list[int],
        encoder: ConfigEncoder,
        minimize: bool = True,  # Default to True (standard for Loss scaling)
    ):
        super().__init__()
        self.scaling_dims = scaling_dims
        self.minimize = minimize
        self.encoder = encoder
        
        n_dims = len(scaling_dims)
        
        # 1. POWERS (The Exponents alpha)
        # Scaling laws usually have negative slopes (Loss decreases as Scale increases)
        # We init around -0.3
        self.register_parameter(
            name="powers", 
            parameter=torch.nn.Parameter(torch.tensor([-0.3] * n_dims, dtype=torch.float64))
        )
        
        # 2. WEIGHTS (The Coefficients A)
        self.register_parameter(
            name="weights", 
            # parameter=torch.nn.Parameter(torch.rand(size=(n_dims,), dtype=torch.float64) * 100)
            parameter=torch.nn.Parameter(torch.tensor([300] * n_dims, dtype=torch.float64))
        )
        
        # 3. BIAS (The Floor / Irreducible Loss)
        self.register_parameter(
            name="bias", 
            parameter=torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print current parameters
        print(f"ScalingMeanModule parameters: powers={self.powers.detach().cpu().numpy()}, weights={self.weights.detach().cpu().numpy()}, bias={self.bias.item()}")

        x_physical = self.encoder.decode_tensor(x)
        x_physical = x_physical[..., self.scaling_dims]
        
        x_physical = x_physical.clamp(min=1e-6)
        x_physical = torch.log(x_physical)

        mean =  torch.sum((x_physical ** self.powers) * self.weights, dim=-1) + self.bias
        return mean
    
def make_default_single_obj_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    encoder: ConfigEncoder,
    *,
    y_transform: OutcomeTransform | None = None,
    objective_minimize: bool = False,
    flop_estimator: Callable[..., int] | None = None,
) -> SingleTaskGP:
    """Default GP for single objective optimization.
    
    Args:
        x: Training input features
        y: Training targets
        encoder: Configuration encoder
        y_transform: Optional outcome transformation
        objective_type: "minimize" or "maximize" (default: "minimize")
        use_huber_loss: Whether to use Huber loss (default: False)
        huber_delta: Huber loss delta parameter (default: 1.0)
    """
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    if y_transform is None:
        y_transform = Standardize(m=1)

    numerics: list[int] = []
    categoricals: list[int] = []
    scaling_dims: list[int] = []
    for hp_name, transformer in encoder.transformers.items():
        if isinstance(transformer, CategoricalToIntegerTransformer):
            categoricals.append(encoder.index_of[hp_name])
        else:
            numerics.append(encoder.index_of[hp_name])
            if getattr(transformer.original_domain, "is_scaling", False):
                scaling_dims.append(encoder.index_of[hp_name])
    
    mean_module = None

    if len(scaling_dims) > 0 and flop_estimator is not None:
        mean_module = ScalingMeanModule(
            scaling_dims=scaling_dims,
            encoder=encoder,
            minimize=objective_minimize,
            # flop_estimator=flop_estimator
        )
        # y = torch.log(y.clamp_min(1e-12))
    # Purely vectorial
    if len(categoricals) == 0:
        return SingleTaskGP(train_X=x, train_Y=y, outcome_transform=y_transform, mean_module=mean_module) # filter just numericals and scaling

    # Purely categorical
    if len(numerics) == 0:
        return SingleTaskGP(
            train_X=x,
            train_Y=y,
            covar_module=default_categorical_kernel(len(categoricals)),
            outcome_transform=y_transform,
            mean_module=None # TODO: add support for categorical
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
        train_X=x, train_Y=y, covar_module=kernel, outcome_transform=y_transform, mean_module=mean_module,
    )


def optimize_acq(
    acq_fn: AcquisitionFunction,
    encoder: ConfigEncoder,
    *,
    n_candidates_required: int = 1,
    num_restarts: int = 20,
    n_intial_start_points: int | None = None,
    acq_options: Mapping[str, Any] | None = None,
    fixed_features: dict[str, Any] | None = None,
    maximum_allowed_categorical_combinations: int = 30,
    hide_warnings: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize the acquisition function."""
    warning_context = (
        disable_warnings(NumericalWarning) if hide_warnings else nullcontext()
    )
    acq_options = acq_options or {}

    _fixed_features: dict[int, float] = {}
    if fixed_features is not None:
        for name, value in fixed_features.items():
            encoded_value = encoder.transformers[name].encode_one(value)
            encoded_index = encoder.index_of[name]
            _fixed_features[encoded_index] = encoded_value

    lower = [domain.lower for domain in encoder.domains]
    upper = [domain.upper for domain in encoder.domains]
    bounds = torch.tensor([lower, upper], dtype=torch.float64)

    cat_transformers = {
        name: t
        for name, t in encoder.transformers.items()
        if (
            name not in _fixed_features  # Don't include those which are fixed by caller
            and t.domain.is_categorical  # Only include categoricals
        )
    }

    # Proceed with regular numerical acquisition
    if not any(cat_transformers):
        # Small heuristic to increase the number of candidates as our
        # dimensionality increases... we apply a cap of 4096,
        # which occurs when len(bounds) >= 8
        # TODO: Need to investigate how num_restarts is fully used in botorch to inform.
        if n_intial_start_points is None:
            n_intial_start_points = min(64 * len(bounds) ** 2, 4096)

        # constraints = acq_options.get("nonlinear_inequality_constraints")
        # if constraints is not None:        
        #     acq_options["ic_generator"] = make_ic_generator(constraints[0][0], encoder)
        
        with warning_context:
            return optimize_acqf(  # type: ignore
                acq_function=acq_fn,
                bounds=bounds,
                q=n_candidates_required,
                num_restarts=num_restarts,
                raw_samples=n_intial_start_points,
                fixed_features=_fixed_features,
                **acq_options,
            )

    # We need to generate the product of all possible combinations of categoricals,
    # first we do a sanity check
    n_combos = reduce(
        lambda x, y: x * y,  # type: ignore
        [t.domain.cardinality for t in cat_transformers.values()],
        1,
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
    # {hp_name: [v1, v2], hp_name2: [v1, v2, v3], ...}
    cats: dict[int, list[float]] = {
        encoder.index_of[name]: [
            float(i)  # NOTE: Botorchs optim requires them to be as floats
            for i in range(transformer.domain.cardinality)  # type: ignore
        ]
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

    # Make sure to include caller's fixed features if provided
    if len(_fixed_features) > 0:
        fixed_cats = [{**cat, **_fixed_features} for cat in fixed_cats]

    with warning_context:
        # TODO: we should deterministically shuffle the fixed_categoricals
        # as the underlying function does not.
        return optimize_acqf_mixed(  # type: ignore
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
    gp: SingleTaskGP,
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
    n_initial_start_points: int = 256,
    maximum_allowed_categorical_combinations: int = 30,
    fixed_acq_features: dict[str, Any] | None = None,
    acq_options: Mapping[str, Any] | None = None,
    hide_warnings: bool = False,
) -> torch.Tensor:
    """Acquire the next configuration to evaluate using a GP.

    Please see the following for:

    * Making a GP to pass in:
        [`make_default_single_obj_gp()`][neps.optimizers.models.gp.make_default_single_obj_gp]
    * Encoding configurations:
        [`encode_trials_for_gp()`][neps.optimizers.models.gp.encode_trials_for_gp]

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
            optimization.
        maximum_allowed_categorical_combinations: The maximum number of categorical
            combinations to allow. If the number of combinations exceeds this, an error
            will be raised.
        acq_options: Additional options to pass to the botorch `optimizer_acqf` function.
        hide_warnings: Whether to hide numerical warnings issued during GP routines.

    Returns:
        The encoded next configuration(s) to evaluate. Use the encoder you provided
        to decode the configuration.
    """
    if seed is not None:
        raise NotImplementedError("Seed is not implemented yet for gps")

    # fit_gpytorch_mll(ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp))

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
        maximum_allowed_categorical_combinations=maximum_allowed_categorical_combinations,
        hide_warnings=hide_warnings,
    )
    return candidates


import torch

class BlackBoxConstraintFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constraint_func, encoder):
        # 1. Save context for backward pass
        ctx.constraint_func = constraint_func
        ctx.encoder = encoder
        ctx.save_for_backward(x)
        
        # 2. Run your original non-differentiable logic
        # We assume standard BoTorch shape: (batch_shape) or (batch, q, d)
        with torch.no_grad():
            # Flatten to handle arbitrary batch shapes safely
            original_shape = x.shape
            x_flat = x.reshape(-1, original_shape[-1])
            
            # Decode using your existing logic
            # Move to CPU for decoding
            conf_list = encoder.decode(x_flat.detach().cpu())
            
            vals = []
            for c in conf_list:
                # Get the value (assuming func returns tuple/list)
                # Ensure this is a FLOAT representing "distance to feasibility"
                val = constraint_func(c)
                
                # IMPORTANT: BoTorch expects Positive = Feasible.
                # If your func returns Negative = Feasible, flip it: val = -val
                vals.append(val)
                
            # Restore shape
            res = torch.tensor(vals, dtype=x.dtype, device=x.device)
            
            # If input was (N, D), output is (N,). If (N, Q, D), output (N, Q)
            if len(original_shape) > 1:
                res = res.view(original_shape[:-1])
                
        return res

    @staticmethod
    def backward(ctx, grad_output):
        # 3. "Fake" the gradient using Finite Differences
        x, = ctx.saved_tensors
        func = ctx.constraint_func
        encoder = ctx.encoder
        
        # A small step size
        epsilon = 1e-3
        
        grad_input = torch.zeros_like(x)
        
        # We must iterate to compute derivatives for every dimension
        # (This can be slow, but it's the only way for black-box funcs)
        x_flat = x.detach().reshape(-1, x.shape[-1])
        grad_out_flat = grad_output.reshape(-1)
        grad_in_flat = grad_input.reshape(-1, x.shape[-1])
        
        for i in range(len(x_flat)):
            # Optimization: If this point doesn't matter for the loss, skip it
            if grad_out_flat[i] == 0:
                continue

            current_val_base = None # Cache if needed
            
            for d in range(x_flat.shape[1]):
                # Create a perturbed point: x + epsilon
                x_p = x_flat[i].clone()
                x_p[d] += epsilon
                
                # Evaluate x + epsilon
                c_p = encoder.decode(x_p.unsqueeze(0).cpu())[0]
                val_p = func(c_p)
                
                # Evaluate x (Center)
                c_base = encoder.decode(x_flat[i].unsqueeze(0).cpu())[0]
                val_base = func(c_base)
                
                # Calculate Slope (Gradient) = (Rise / Run)
                slope = (val_p - val_base) / epsilon
                
                # Chain Rule: Gradient = Slope * Incoming_Gradient
                grad_in_flat[i, d] = slope * grad_out_flat[i]
                
        return grad_input.view_as(x), None, None


def encode_constraints_func(
    constraints_func,
    *,
    encoder,
    device=None,
):
    def inner(x):
        # Use .apply() to call the autograd function
        return BlackBoxConstraintFn.apply(x, constraints_func, encoder)
    return inner


def make_ic_generator(constraint_func, encoder):
    """
    Creates an initial condition generator that respects non-linear constraints.
    Uses the custom Sobol sampler from NePS.
    """
    def ic_generator(acq_function, bounds, q, num_restarts, raw_samples, fixed_features=None, **kwargs):
        # 1. Initialize the custom Sobol sampler
        ndim = bounds.shape[1]
        sampler = Sobol(ndim=ndim, scramble=True)
        
        # 2. Sample raw candidates
        # We need 'raw_samples' starting points.
        # If q > 1, each starting point is actually a batch of q candidates.
        # Shape needed: (raw_samples, q, ndim)
        sample_shape = torch.Size([raw_samples, q])
        
        # Note: We pass 'encoder' as the target domain ('to'). 
        # Ensure your Domain.translate supports ConfigEncoder as 'to'.
        X_cand = sampler.sample(
            n=sample_shape, 
            to=encoder, 
            device=bounds.device,
            dtype=bounds.dtype
        )

        # 3. Filter using the constraint function
        with torch.no_grad():
            # constraint_func returns >= 0 for valid
            # X_cand shape: (raw_samples, q, d)
            # We flatten to (raw_samples * q, d) if the constraint func expects 2D, 
            # or pass 3D if it handles it. 
            # Your encode_constraints_func wrapper handles dimensions, so we pass as is.
            constraint_vals = constraint_func(X_cand)
            
            # Constraint satisfied if >= 0
            valid_mask = (constraint_vals >= 0)
            
            # If q > 1, the constraint returns shape (raw_samples, q).
            # A starting point is only valid if ALL q candidates in it are valid.
            if valid_mask.ndim > 1:
                valid_mask = valid_mask.all(dim=-1)
                
            X_valid = X_cand[valid_mask]

        # 4. Handle Insufficient Valid Points (Fallback)
        # If strict constraints leave us with fewer points than 'num_restarts',
        # we must fill the gap to prevent the optimizer from crashing.
        if len(X_valid) < num_restarts:
            logger.warning(
                f"Constraint is too strict: found {len(X_valid)} valid points out of {raw_samples}. "
                "Optimization performance may degrade."
            )
            
            if len(X_valid) == 0:
                # Emergency: Return the raw candidates even if invalid, 
                # effectively falling back to standard behavior.
                return X_cand[:num_restarts]
                
            # Recycle valid points to fill the quota
            needed = num_restarts - len(X_valid)
            repeats = (needed // len(X_valid)) + 2
            X_valid = torch.cat([X_valid] * repeats)[:num_restarts]

        # 5. Select Best Starts
        # Evaluate the acquisition function on the valid points to pick the best starters.
        with torch.no_grad():
            acq_vals = acq_function(X_valid)
        
        # ROBUST NaN HANDLING: Replace NaN acquisition values with minimum valid value
        nan_mask = torch.isnan(acq_vals)
        if nan_mask.any():
            # Get valid (non-NaN) values
            valid_acq_vals = acq_vals[~nan_mask]
            
            if len(valid_acq_vals) > 0:
                # Use minimum valid value for NaNs (conservative choice)
                fill_value = valid_acq_vals.min().item()
            else:
                # All values are NaN - use 0 as fallback
                fill_value = 0.0
            
            acq_vals[nan_mask] = fill_value
            logger.warning(
                f"NaN detected in acquisition function evaluation during IC generation: "
                f"{nan_mask.sum().item()} NaN values replaced with {fill_value:.6f}"
            )
        
        # We want the indices of the highest acquisition values
        _, best_idxs = torch.topk(acq_vals, min(num_restarts, len(X_valid)))
        
        best_ics = X_valid[best_idxs]
        
        # ROBUST NaN HANDLING: Check if initial conditions themselves have NaN values
        if torch.isnan(best_ics).any():
            nan_positions = torch.isnan(best_ics)
            logger.warning(
                f"NaN detected in initial conditions: {nan_positions.any(dim=0).sum().item()} "
                f"dimensions affected. Replacing with bounds midpoint."
            )
            # Replace NaN values with bounds midpoint
            bounds_mid = (bounds[0] + bounds[1]) / 2
            for dim in range(best_ics.shape[-1]):
                if nan_positions[:, dim].any():
                    best_ics[nan_positions[:, dim], dim] = bounds_mid[dim]
        
        return best_ics

    return ic_generator