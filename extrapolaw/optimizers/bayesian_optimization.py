from __future__ import annotations

import copy
import itertools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import torch
from botorch.acquisition import LinearMCObjective
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.utils.sampling import draw_sobol_samples

from extrapolaw.exceptions import BudgetExhaustedError, ConstraintViolationError
from extrapolaw.optimizers.models.gp import (
    LogCost,
    encode_constraints_func,
    encode_trials_for_gp,
    fit_and_acquire_from_gp,
    make_default_single_obj_gp,
    make_single_obj_gp,
)
from extrapolaw.optimizers.optimizer import (
    ImportedConfig,
    SampledConfig,
    ScalingOptimizer,
)
from extrapolaw.optimizers.utils.initial_design import make_initial_design
from extrapolaw.optimizers.utils.util import _get_next_trial_id
from extrapolaw.space.extrapolaw_spaces.extrapolaw_space import (
    convert_extrapolaw_to_classic_search_space,
)
from extrapolaw.space.extrapolaw_spaces.parameters import PipelineSpace

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP

    from extrapolaw.space import ConfigEncoder, SearchSpace
    from extrapolaw.state import BudgetInfo, Trial
    from extrapolaw.state.pipeline_eval import UserResultDict
    from extrapolaw.utils.cost_estimator import CostEstimator

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.ERROR)  # Suppress matplotlib debug logs

INITIAL_DESIGN_OVERSAMPLING_FACTOR = 40

#: The fraction of the cost caps the acquisition is optimized below.
ACQUISITION_COST_MARGIN = 1e-3


@dataclass
class BayesianOptimization(ScalingOptimizer):
    """Uses `botorch` as an engine for doing bayesian optimiziation."""

    space: SearchSpace | PipelineSpace
    """The search space to use."""

    encoder: ConfigEncoder
    """The encoder to use for encoding and decoding configurations."""

    sample_prior_first: bool
    """Whether to sample the prior configuration first."""

    cost_aware: bool | Literal["log"]
    """Whether to consider the cost of configurations in decision making."""

    n_initial_design: int
    """The number of initial design samples to use before fitting the GP."""

    device: torch.device | None
    """The device to use for the optimization."""

    num_restarts: int = 20
    """The number of restarts to use when optimizing the acquisition function.

    Lowering this trades acquisition quality for a faster point acquisition.
    """

    n_initial_start_points: int | None = 256
    """The number of initial start points to use when optimizing the acquisition
    function.

    If `None`, a dimension-dependent heuristic is used instead. Lowering this
    trades acquisition quality for its speed.
    """

    constraints_func: Callable[[Mapping[str, Any]], float] | None = None

    acquisition_func_type: Literal["EI", "IT-JES", "IT-MES", "IT-SCAN-JES"] = "EI"
    cost_estimator: CostEstimator | None = None
    target_flops: int | None = None
    """The FLOPs the `"IT-SCAN-JES"` acquisition samples the optima around, used by
    it only. It may lie beyond `max_cost_per_config`, the max FLOPs per config, to
    extrapolate to."""

    log_y: bool = True
    """Whether the GP with the power law prior mean in the cost models the log of the
    objective."""

    prior_mean: Literal["power_law", "zero"] = "power_law"
    """The prior mean of the GP. `"power_law"` is the power law in the cost fitted on the
    trials, which needs a `cost_estimator`, else the GP falls back to the learned
    constant mean of `SingleTaskGP`. `"zero"` ablates it with a zero mean of the
    standardized outcomes, the `cost_estimator` still serving the budget schedule and
    the max cost per config."""

    budget_schedule: Literal["none", "space_expansion", "left_budget_contraction"] = (
        "none"
    )
    """How the max cost of a sampled config follows the spent cost budget.

    * `"none"`: only `max_cost_per_config` caps it.
    * `"space_expansion"`: a ladder of caps. The initial design gets `B / 2^L` of the
      budget `B`, split evenly over its configs, and level `i` then allows configs up to
      `B / 2^(L - i)`, with `L = ceil(log2(B / max_cost_per_config))`. A level whose
      cap leaves no feasible config is skipped.
    * `"left_budget_contraction"`: the initial design configs cost up to
      `B / (10 * n_initial_design)`, later ones up to the budget left.

    Both schedules need a `cost_estimator` and a cost budget given to
    `extrapolaw.acquire()`, and are clamped at `max_cost_per_config` and the budget left.
    Once the budget left is below the cap of the initial design configs, the cheapest
    the schedule samples, or no feasible config fits in it, sampling raises
    `BudgetExhaustedError` and the worker stops.
    """

    def get_constraint_func(
        self, max_cost: float | None = None, *, margin: float = 0.0
    ) -> Callable[[Mapping[str, Any]], float] | None:
        """The constraint of the sampled configs, with `max_cost` the cap of this round
        of the `budget_schedule`. With a `margin`, the configs cost at most
        `(1 - margin)` of `max_cost` and of `max_cost_per_config`, the user constraint
        is unchanged.
        """
        assert not isinstance(self.space, PipelineSpace)  # Converted in __post_init__
        constraints_func = self.constraints_func
        if max_cost is not None:
            assert self.cost_estimator is not None
            estimator, user_constraint = self.cost_estimator, self.constraints_func

            def constraints_func(config: Mapping[str, Any]) -> float:
                # Normalized like `budget_constraint()`, `config` has the constants
                slack = 1.0 - estimator(**config) / max_cost - margin
                if user_constraint is None:
                    return slack
                return min(slack, user_constraint(config))

        # The GP only ever sees the searchables, the cost estimator needs the constants
        return self.budget_constraint(
            constraints_func, constants=self.space.constants, margin=margin
        )

    def _total_budget(self, budget_info: BudgetInfo | None) -> float:
        budget = None
        if budget_info is not None:
            budget = budget_info.total_cost_to_spend
            if budget is None:
                budget = budget_info.worker_cost_to_spend
        if budget is None:
            raise ValueError(
                f"The {self.budget_schedule!r} budget schedule needs a cost budget,"
                " pass `total_cost_to_spend` or `worker_cost_to_spend` to"
                " `extrapolaw.acquire()`."
            )
        return budget

    def _schedule_bounds(self, total: float) -> tuple[int, float]:
        """The number of levels `L` of the space expansion and the budget of its
        initial design.
        """
        top = self.max_cost_per_config
        if top is None:
            raise ValueError(
                "The 'space_expansion' budget schedule needs `max_cost_per_config` to be"
                " given to `extrapolaw.acquire()`."
            )
        max_expansion = 1 if total <= top else max(1, math.ceil(math.log2(total / top)))
        return max_expansion, total / 2**max_expansion

    def _expansion_level(self, spent: float, total: float) -> int:
        """The level of the space expansion after spending `spent` of `total`.

        Level 0 is the initial design, level `i >= 1` has a budget of
        `total / 2^(L - i)`.
        """
        max_expansion, initial_budget = self._schedule_bounds(total)
        if spent < initial_budget:
            return 0

        spent_in_expansion = spent - initial_budget
        for level in range(1, max_expansion + 1):
            budget_for_level = total / 2 ** (max_expansion - level)
            if spent_in_expansion < budget_for_level:
                return level
            spent_in_expansion -= budget_for_level
        return max_expansion

    def _schedule_cap(
        self,
        spent: float,
        total: float,
        *,
        is_initial_design: bool,
        expand: int = 0,
    ) -> float:
        """The max cost of a config sampled this round, `expand` levels above the
        current one of the space expansion.
        """
        if self.budget_schedule == "left_budget_contraction":
            if is_initial_design:
                cap = total / (10 * self.n_initial_design)
            else:
                cap = total - spent
        else:
            max_expansion, initial_budget = self._schedule_bounds(total)
            level = self._expansion_level(spent, total) + expand
            if level == 0 and spent < initial_budget:
                cap = (
                    initial_budget / self.n_initial_design
                    if is_initial_design
                    else initial_budget - spent
                )
            else:
                cap = total / 2 ** (max_expansion - level)
        if self.max_cost_per_config is not None:
            cap = min(cap, self.max_cost_per_config)
        return min(cap, total - spent)

    def _min_schedule_cap(self, total: float) -> float:
        """The cap of the initial design configs, the cheapest of the schedule."""
        if self.budget_schedule == "left_budget_contraction":
            cap = total / (10 * self.n_initial_design)
        else:
            _, initial_budget = self._schedule_bounds(total)
            cap = initial_budget / self.n_initial_design
        if self.max_cost_per_config is not None:
            cap = min(cap, self.max_cost_per_config)
        return cap

    def _cost_estimator_with_constants(self) -> Callable[..., float] | None:
        if self.cost_estimator is None:
            return None
        assert not isinstance(self.space, PipelineSpace)  # Converted in __post_init__
        estimator, constants = self.cost_estimator, self.space.constants
        return lambda **config: estimator(**{**constants, **config})

    def __post_init__(self) -> None:
        if isinstance(self.space, PipelineSpace):
            converted_space = convert_extrapolaw_to_classic_search_space(self.space)
            if converted_space is not None:
                self.space = converted_space
            else:
                raise ValueError(
                    "This optimizer only supports HPO search spaces, please use an"
                    " Extrapolaw"
                    " space-compatible optimizer."
                )
        self.design_samples: list[dict[str, Any]] | None = None

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        return self.sample_candidates(trials, budget_info, n=n)

    def _plot_extrapolation_landscape(
        self,
        gp: SingleTaskGP,
        best_input_tensor: torch.Tensor,
        bounds: torch.Tensor,
    ) -> plt.Figure | None:
        """Generate extrapolation landscape visualization.

        Args:
            gp: The trained Gaussian Process model.
            best_input_tensor: The optimal input tensor found via BO.
            bounds: The parameter bounds tensor.

        Returns:
            matplotlib Figure object or None if visualization fails.
        """
        try:
            # Select dimensions to plot
            if (
                hasattr(self.encoder, "numeric_dims")
                and len(self.encoder.numeric_dims) >= 2
            ):
                dim_x, dim_y = self.encoder.numeric_dims[0], self.encoder.numeric_dims[1]
            else:
                dim_x, dim_y = 0, 1

            # A. Prepare training data
            x_cpu = gp.train_inputs[0].detach().cpu()
            y_cpu = gp.train_targets.detach().cpu()

            # B. Generate meshgrid
            n_grid = 50
            bounds_cpu = bounds.detach().cpu()

            x_vals = torch.linspace(bounds_cpu[0, dim_x], bounds_cpu[1, dim_x], n_grid)
            y_vals = torch.linspace(bounds_cpu[0, dim_y], bounds_cpu[1, dim_y], n_grid)
            gx, gy = torch.meshgrid(x_vals, y_vals, indexing="xy")

            # Flatten grid
            grid_flat = torch.zeros(
                (n_grid * n_grid, bounds_cpu.shape[1]),
                dtype=best_input_tensor.dtype,
                device=best_input_tensor.device,
            )

            grid_flat[:] = best_input_tensor.detach()
            grid_flat[:, dim_x] = gx.flatten()
            grid_flat[:, dim_y] = gy.flatten()

            # Predict with NaN robustness
            with torch.no_grad():
                posterior_grid = gp.posterior(grid_flat)
                mean_grid = posterior_grid.mean.view(n_grid, n_grid)

                # ROBUST NaN HANDLING: Replace any NaN values in grid predictions
                if torch.isnan(mean_grid).any():
                    y_train = gp.train_targets
                    fill_value = torch.nanmedian(y_train).item()
                    nan_mask = torch.isnan(mean_grid)
                    mean_grid[nan_mask] = fill_value
                    logger.warning(
                        "NaN values detected in extrapolation grid "
                        f"({nan_mask.sum().item()} values). "
                        f"Replaced with training median ({fill_value:.6f})."
                    )

            # C. Plot
            fig, ax = plt.subplots(figsize=(10, 7))

            c = ax.contourf(
                gx.cpu().numpy(),
                gy.cpu().numpy(),
                mean_grid.cpu().numpy(),
                levels=25,
                cmap="viridis_r",
                alpha=0.8,
            )
            cbar = plt.colorbar(c)
            cbar.set_label("Predicted Loss (Mean)", rotation=270, labelpad=15)

            ax.scatter(
                x_cpu[:, dim_x].numpy(),
                x_cpu[:, dim_y].numpy(),
                c=y_cpu.numpy(),
                cmap="viridis_r",
                edgecolors="k",
                s=50,
                label="Observed Data",
            )

            ax.scatter(
                best_input_tensor[dim_x].detach().cpu().numpy(),
                best_input_tensor[dim_y].detach().cpu().numpy(),
                c="red",
                marker="*",
                s=200,
                label="Extrapolated Optimum",
            )

            ax.set_xlabel(f"Dimension {dim_x}")
            ax.set_ylabel(f"Dimension {dim_y}")
            ax.set_title("GP Extrapolation Landscape")
            ax.legend()

            plt.tight_layout()
            return fig

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to generate extrapolation landscape: {e}")
            return None

    def sample_candidates(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        if self.budget_schedule == "none":
            return self._sample_candidates(trials, budget_info, n)

        if self.cost_estimator is None:
            raise ValueError(
                f"The {self.budget_schedule!r} budget schedule needs a `cost_estimator`"
                " to be given to `extrapolaw.acquire()`."
            )
        total = self._total_budget(budget_info)
        # Pending trials count too, so parallel workers see each other's spend
        spent = sum(self.cost_estimator(**trial.config) for trial in trials.values())
        is_initial_design = len(trials) < self.n_initial_design
        cost_bounds = (
            self._min_schedule_cap(total),
            self.max_cost_per_config or total,
        )

        left = total - spent
        if left < cost_bounds[0]:
            raise BudgetExhaustedError(
                f"Only {left:e} of the cost budget {total:e} is left, less than the"
                f" {cost_bounds[0]:e} of the cheapest config the schedule samples."
            )

        max_cost = self._schedule_cap(spent, total, is_initial_design=is_initial_design)
        logger.info(
            f"Budget schedule {self.budget_schedule!r}: max cost {max_cost:e} per"
            f" config, spent {spent:e} of {total:e}."
        )
        try:
            return self._sample_candidates(
                trials, budget_info, n, max_cost=max_cost, cost_bounds=cost_bounds
            )
        except ConstraintViolationError as e:
            error = e

        # A level without feasible configs moves on to the next one, unless the next
        # one allows no more, as the budget left or `max_cost_per_config` limit it
        next_cost = self._schedule_cap(spent, total, is_initial_design=False, expand=1)
        if self.budget_schedule == "space_expansion" and next_cost > max_cost:
            logger.warning(
                f"No feasible config under the max cost of this level ({error}),"
                f" retrying with the next level's {next_cost:e}."
            )
            try:
                return self._sample_candidates(
                    trials, budget_info, n, max_cost=next_cost, cost_bounds=cost_bounds
                )
            except ConstraintViolationError as e:
                error, max_cost = e, next_cost

        if max_cost >= left:
            raise BudgetExhaustedError(
                f"No feasible config costs at most the {left:e} of the cost budget"
                f" {total:e} left."
            ) from error
        raise error

    def _sample_candidates(  # noqa: C901, PLR0912, PLR0915
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
        *,
        max_cost: float | None = None,
        cost_bounds: tuple[float, float] | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        """Sample the configs with a cost up to `max_cost`, and `cost_bounds` the
        smallest and largest cost the schedule samples, to normalize the cost term of the
        acquisition with.
        """
        assert not isinstance(self.space, PipelineSpace)  # Converted in __post_init__
        # If fidelities exist, sample from them as normal
        # This is a bit of a hack, as we set them to max fidelity
        # afterwards, but we need the complete space to sample
        if self.space.fidelity is not None:
            parameters = {**self.space.searchables, **self.space.fidelities}
        else:
            parameters = {**self.space.searchables}

        list(parameters.keys())

        n_to_sample = 1 if n is None else n

        # Sampled configs are always run at the max fidelity, so that is also what
        # the constraint has to be checked against.
        fidelity_upper = (
            {key: value.upper for key, value in self.space.fidelities.items()}
            if self.space.fidelity is not None
            else {}
        )

        def _at_max_fidelity(
            func: Callable[[Mapping[str, Any]], float] | None,
        ) -> Callable[[Mapping[str, Any]], float] | None:
            if func is None or not fidelity_upper:
                return func
            return lambda config: func({**config, **fidelity_upper})

        constraints_func = _at_max_fidelity(self.get_constraint_func(max_cost))
        # The acquisition is optimized a margin below the cost caps, as its optimum
        # often lies on one and rounding on decode would then push it over, the config
        # being replaced by a random one. The acquired configs are still checked
        # against the caps themselves.
        acquisition_constraints_func = _at_max_fidelity(
            self.get_constraint_func(max_cost, margin=ACQUISITION_COST_MARGIN)
        )
        cost_estimator = self._cost_estimator_with_constants()

        n_sampled = len(trials)

        next_id = _get_next_trial_id(trials)
        id_generator = iter(str(i) for i in itertools.count(next_id))

        # If the amount of configs evaluated is less than the initial design
        # requirement, keep drawing from initial design
        n_evaluated = sum(
            1
            for trial in trials.values()
            if (
                trial.report is not None
                and trial.report.objective_to_minimize is not None
            )
            or (trial.metadata.state == "evaluating")
        )
        sampled_configs: list[SampledConfig] = []

        if n_evaluated < self.n_initial_design:
            # For reproducibility, we need to ensure we do the same sample of all
            # configs each time.
            if (
                self.design_samples is not None
                and len(self.design_samples) >= n_evaluated + n_to_sample
            ):
                pass
            else:
                self.design_samples = make_initial_design(
                    parameters=parameters,
                    encoder=self.encoder,
                    sample_prior_first=self.sample_prior_first
                    if n_sampled == 0
                    else False,
                    sampler="uniform",
                    seed=None,  # TODO: Seeding, but need to avoid repeating configs
                    sample_size=self.n_initial_design,
                    constraints_func=constraints_func,
                    oversampling_factor=INITIAL_DESIGN_OVERSAMPLING_FACTOR,
                )

            # Then take the subset we actually need
            design_samples = self.design_samples[n_evaluated:]
            for sample in design_samples:
                sample.update(self.space.constants)
                if self.space.fidelity is not None:
                    sample.update(
                        {key: value.upper for key, value in self.space.fidelities.items()}
                    )

            sampled_configs.extend(
                [
                    SampledConfig(id=config_id, config=config)
                    for config_id, config in zip(
                        id_generator,
                        design_samples,
                        # NOTE: We use a generator for the ids so no need for strict
                        strict=False,
                    )
                ]
            )

            if len(sampled_configs) >= n_to_sample:
                return sampled_configs[0] if n is None else sampled_configs

        # Otherwise, we encode trials and setup to fit and acquire from a GP
        data, encoder = encode_trials_for_gp(
            trials,
            parameters,
            device=self.device,
            encoder=self.encoder,
        )
        encoded_constraints_func = (
            encode_constraints_func(constraints_func, encoder=encoder)
            if constraints_func is not None
            else None
        )

        cost_percent = None
        if self.cost_aware:
            # TODO: Interaction with `"log"` cost aware
            if self.cost_aware == "log":
                raise NotImplementedError("Log cost aware not implemented yet.")

            if budget_info is None:
                raise ValueError(
                    "Must provide a 'cost' to configurations if using cost"
                    " with BayesianOptimization."
                )
            if budget_info.worker_cost_to_spend is None:
                raise ValueError("Cost budget must be set if using cost")
            cost_percent = budget_info.used_cost_budget / budget_info.worker_cost_to_spend

        n_to_acquire = n_to_sample - len(sampled_configs)

        if self.prior_mean == "zero":
            gp = make_single_obj_gp(
                x=data.x,
                y=data.y,
                encoder=encoder,
                prior_mean="zero",
                log_y=self.log_y,
            )
        elif cost_estimator is None:
            logger.warning(
                "No cost estimator, the GP falls back to a constant prior mean"
                " instead of the power law in the cost."
            )
            gp = make_default_single_obj_gp(x=data.x, y=data.y, encoder=encoder)
        else:
            # Fit a power law to determine GP mean prior
            from extrapolaw.utils.power_law import fit_power_law_mean_params

            completed = [
                (trial.config, trial.report.objective_to_minimize)
                for trial in trials.values()
                if trial.report is not None
                and trial.report.objective_to_minimize is not None
            ]
            mean_params = fit_power_law_mean_params(
                [config for config, _ in completed],
                [objective for _, objective in completed],
                cost_estimator,
            )
            if mean_params.fit is not None:
                logger.info(
                    f"Fitted power law: slope={mean_params.slope:.4f}, "
                    f"intercept={mean_params.intercept:.4f}, "
                    f"r²={mean_params.fit.r_squared:.4f}, "
                    f"n={mean_params.fit.num_points}"
                )
            else:
                logger.warning(
                    "Insufficient data for power law fit "
                    f"({len(mean_params.flops)} trials). "
                    f"Using default slope={mean_params.slope}, "
                    f"intercept={mean_params.intercept}."
                )

            gp = make_single_obj_gp(
                x=data.x,
                y=data.y,
                encoder=encoder,
                prior_mean="power_law",
                cost_estimator=cost_estimator,
                min_cost=mean_params.min_flops,
                mean_slope=mean_params.slope,
                mean_bias=mean_params.intercept,
                log_y=self.log_y,
            )
        from botorch.fit import fit_gpytorch_mll
        from gpytorch import ExactMarginalLogLikelihood

        fit_gpytorch_mll(ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp))

        if self.acquisition_func_type == "EI":
            acquisition = qLogNoisyExpectedImprovement(
                model=gp,
                X_baseline=data.x,
                # Unfortunatly, there's no option to indicate that we minimize
                # the AcqFunction so we need to do some kind of transformation.
                # https://github.com/pytorch/botorch/issues/2316#issuecomment-2085964607
                objective=LinearMCObjective(weights=torch.tensor([-1.0])),
                X_pending=data.x_pending,
                prune_baseline=True,
            )
        elif self.acquisition_func_type == "IT-JES":
            from botorch.acquisition.joint_entropy_search import qJointEntropySearch
            from botorch.acquisition.utils import get_optimal_samples

            lower = [domain.lower for domain in encoder.domains]
            upper = [domain.upper for domain in encoder.domains]
            bounds = torch.tensor(
                [lower, upper], dtype=torch.float64, device=data.x.device
            )

            if encoded_constraints_func is None:
                optimal_inputs, optimal_outputs = get_optimal_samples(
                    model=gp, bounds=bounds, num_optima=4, maximize=False
                )
            else:
                # `get_optimal_samples` can't handle the constraints, so we
                # sample the optima over a feasible candidate set instead
                candidate_set = _feasible_candidates(
                    draw_sobol_samples(bounds=bounds, n=2048, q=1).squeeze(1),
                    encoded_constraints_func,
                )
                optimal_inputs, optimal_outputs = _sample_optima(
                    gp, candidate_set, num_optima=4
                )

            # Check for NaN/inf in optimal samples and fallback to EI if found
            if (
                torch.isnan(optimal_inputs).any()
                or torch.isinf(optimal_inputs).any()
                or torch.isnan(optimal_outputs).any()
                or torch.isinf(optimal_outputs).any()
            ):
                raise ValueError(
                    "NaN/Inf detected in optimal_samples from get_optimal_samples. "
                )

            acquisition = qJointEntropySearch(
                model=gp,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                condition_noiseless=True,
                estimation_type="MC",
                num_samples=128,
                X_pending=None,
                maximize=False,
            )
        elif self.acquisition_func_type == "IT-MES":
            from botorch.acquisition.max_value_entropy_search import (
                qLowerBoundMaxValueEntropy,
            )

            lower = [domain.lower for domain in encoder.domains]
            upper = [domain.upper for domain in encoder.domains]
            bounds = torch.tensor(
                [lower, upper], dtype=torch.float64, device=data.x.device
            )

            candidate_set = (
                draw_sobol_samples(bounds=bounds, n=512, q=1)
                .squeeze(1)
                .to(dtype=torch.float64)
            )
            candidate_set = _feasible_candidates(candidate_set, encoded_constraints_func)

            acquisition = qLowerBoundMaxValueEntropy(
                model=gp,
                candidate_set=candidate_set,
                maximize=False,
            )
        elif self.acquisition_func_type == "IT-SCAN-JES":
            from extrapolaw.optimizers.acquisition.information_theoretic import (
                CostWeightedModifiedJES,
            )

            lower = [domain.lower for domain in encoder.domains]
            upper = [domain.upper for domain in encoder.domains]
            bounds = torch.tensor(
                [lower, upper], dtype=torch.float64, device=data.x.device
            )

            # Build a candidate set restricted to configs near C_target. The optima
            # at the target are never evaluated, so the pool only keeps the user
            # constraint, not the cap of the budget schedule or
            # `max_cost_per_config`, which the target usually lies beyond. The
            # acquisition itself is still optimized under both.
            raw_candidates = (
                draw_sobol_samples(bounds=bounds, n=2048, q=1)
                .squeeze(1)
                .to(dtype=torch.float64)
            )
            user_constraint = self.constraints_func
            if user_constraint is not None:
                constants = {**self.space.constants, **fidelity_upper}
                raw_candidates = _feasible_candidates(
                    raw_candidates,
                    encode_constraints_func(
                        lambda config: user_constraint({**constants, **config}),
                        encoder=encoder,
                    ),
                )
            target_flops = self.target_flops
            if target_flops is None:
                raise ValueError(
                    "The 'IT-SCAN-JES' acquisition needs `target_flops`, the FLOPs"
                    " it samples the optima around."
                )
            assert cost_estimator is not None
            log_cost = LogCost(encoder, cost_estimator)
            log10_target = math.log10(target_flops)
            distance = (log_cost.evaluate(raw_candidates) - log10_target).abs()
            # Widen the window around the target FLOPs until it holds enough
            # candidates, starting at 2/3 to 1.5x the target
            for factor in (1.5, 3.0, 10.0):
                in_window = distance < math.log10(factor)
                if in_window.sum() >= 4:
                    if factor > 1.5:
                        logger.warning(
                            "Too few candidates cost within 1.5x of the target"
                            f" FLOPs {target_flops:e}, sampling the optima over"
                            f" those within {factor}x."
                        )
                    break
            else:
                logger.warning(
                    "No window around the target FLOPs holds enough candidates,"
                    " sampling the optima over all candidates."
                )
                in_window = torch.ones_like(in_window)
            candidate_set = raw_candidates[in_window]

            # Sample S posterior paths over the C_target-filtered candidate set
            # and take the argmin of each path to get fantasy optima at C_target.
            optimal_inputs, optimal_outputs = _sample_optima(
                gp, candidate_set, num_optima=4
            )

            if torch.isnan(optimal_inputs).any() or torch.isnan(optimal_outputs).any():
                raise ValueError("NaN in modified-JES optimal samples")

            # Fixed bounds for the normalized cost term, so it doesn't move with
            # the costs observed so far. The lower one is the cheapest config of the
            # space, not the cheapest the schedule samples: cheaper configs stay
            # feasible after the initial design, and below the lower bound the cost
            # term clamps to its epsilon, which would favour all of them alike
            if cost_bounds is None:
                top_cost = self.max_cost_per_config or float(
                    10 ** log_cost.evaluate(raw_candidates).max()
                )
            else:
                top_cost = cost_bounds[1]
            lowest_log_cost = torch.cat(
                [log_cost.evaluate(bounds[:1]), log_cost.evaluate(raw_candidates)]
            ).min()
            cost_bounds = (float(10**lowest_log_cost), top_cost)
            acquisition = CostWeightedModifiedJES(
                model=gp,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                log_cost=log_cost,
                c_min_log=math.log10(cost_bounds[0]),
                c_max_log=math.log10(cost_bounds[1]),
            )
        else:
            raise NotImplementedError(
                "Acquisition function not supported for single objective optimization."
            )

        # Prepare acq_options with constraints if provided
        acq_opts = {}
        if acquisition_constraints_func is not None:
            acq_opts["nonlinear_inequality_constraints"] = [
                (
                    encode_constraints_func(
                        acquisition_constraints_func, encoder=encoder
                    ),
                    True,
                )
            ]

        logger.info("Start Gaussian Process fitting and acquisition...")
        candidates = fit_and_acquire_from_gp(
            gp=gp,
            x_train=data.x,
            encoder=encoder,
            acquisition=acquisition,
            n_candidates_required=n_to_acquire,
            costs=data.cost
            if (
                self.cost_aware is not False
                and self.acquisition_func_type != "IT-SCAN-JES"
            )
            else None,
            cost_percentage_used=cost_percent,
            costs_on_log_scale=self.cost_aware == "log",
            num_restarts=self.num_restarts,
            n_initial_start_points=self.n_initial_start_points,
            fixed_acq_features=fidelity_upper or None,
            acq_options=acq_opts,
            hide_warnings=True,
        )

        configs = encoder.decode(candidates)

        # Rounding on decode, the discrete optimizers or a failed SLSQP run can
        # still give infeasible configs, replace those with random feasible ones.
        if constraints_func is not None:
            infeasible = [i for i, c in enumerate(configs) if constraints_func(c) < 0]
            if infeasible:
                logger.warning(
                    f"{len(infeasible)} acquired config(s) violate the constraints,"
                    " replacing them with random feasible configs."
                )
                replacements = make_initial_design(
                    parameters=parameters,
                    encoder=encoder,
                    sampler="uniform",
                    sample_size=len(infeasible),
                    sample_prior_first=False,
                    constraints_func=constraints_func,
                    oversampling_factor=INITIAL_DESIGN_OVERSAMPLING_FACTOR,
                )
                for i, replacement in zip(infeasible, replacements, strict=True):
                    configs[i] = replacement

        for config in configs:
            config.update(self.space.constants)
            config.update(fidelity_upper)

        sampled_configs.extend(
            [
                SampledConfig(id=config_id, config=config)
                for config_id, config in zip(
                    id_generator,
                    configs,
                    # NOTE: We use a generator for the ids so no need for strict
                    strict=False,
                )
            ]
        )
        return sampled_configs[0] if n is None else sampled_configs

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        first_id = _get_next_trial_id(trials)
        return [
            ImportedConfig(
                id=str(i),
                config=copy.deepcopy(config),
                result=copy.deepcopy(result),
            )
            for i, (config, result) in enumerate(external_evaluations, start=first_id)
        ]


def _feasible_candidates(
    candidates: torch.Tensor,
    encoded_constraints_func: Callable[[torch.Tensor], torch.Tensor] | None,
    *,
    min_count: int = 4,
) -> torch.Tensor:
    """Keep only the (encoded) candidates that satisfy the constraints.

    Falls back to all candidates if fewer than `min_count` are feasible.
    """
    if encoded_constraints_func is None:
        return candidates

    with torch.no_grad():
        feasible = encoded_constraints_func(candidates) >= 0

    if feasible.sum() < min_count:
        logger.warning(
            f"Only {int(feasible.sum())}/{len(candidates)} candidates are feasible,"
            " ignoring the constraints for the candidate set."
        )
        return candidates
    return candidates[feasible]


def _sample_optima(
    gp: SingleTaskGP,
    candidate_set: torch.Tensor,
    num_optima: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `num_optima` posterior paths over the candidate set and take the
    argmin of each path as a sampled optimum.

    Returns:
        The optimal inputs `(num_optima, d)` and outputs `(num_optima, 1)`.
    """
    posterior_at_candidates = gp.posterior(candidate_set.unsqueeze(-2))
    # samples: (num_optima, N, 1, 1) → (num_optima, N)
    samples = (
        posterior_at_candidates.rsample(torch.Size([num_optima])).squeeze(-1).squeeze(-1)
    )
    best_idxs = samples.argmin(dim=-1)  # (num_optima,)
    optimal_inputs = candidate_set[best_idxs]  # (S, D)
    optimal_outputs = samples[torch.arange(num_optima), best_idxs].unsqueeze(-1)  # (S, 1)
    return optimal_inputs, optimal_outputs
