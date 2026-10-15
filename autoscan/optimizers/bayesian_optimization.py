from __future__ import annotations

import copy
import itertools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import LinearMCObjective
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import draw_sobol_samples

from autoscan.optimizers.models.gp import (
    encode_constraints_func,
    encode_trials_for_gp,
    fit_and_acquire_from_gp,
    make_default_single_obj_gp,
)
from autoscan.optimizers.optimizer import (
    ImportedConfig,
    SampledConfig,
    ScalingOptimizer,
)
from autoscan.optimizers.utils.initial_design import make_initial_design
from autoscan.optimizers.utils.util import _get_next_trial_id
from autoscan.space.autoscan_spaces.autoscan_space import (
    convert_autoscan_to_classic_search_space,
)
from autoscan.space.autoscan_spaces.parameters import PipelineSpace

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP

    from autoscan.sampling import Prior
    from autoscan.space import ConfigEncoder, SearchSpace
    from autoscan.state import BudgetInfo, Trial
    from autoscan.state.pipeline_eval import UserResultDict
    from autoscan.utils.cost_estimator import CostEstimator

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.ERROR)  # Suppress matplotlib debug logs

INITIAL_DESIGN_OVERSAMPLING_FACTOR = 40


def _pibo_exp_term(
    n_sampled_already: int,
    ndims: int,
    initial_design_size: int,
) -> float:
    # pibo paper
    # https://arxiv.org/pdf/2204.11051
    #
    # they use some constant determined from max problem budget. seems impractical,
    # given we might not know the final budget (i.e. imagine you iteratively increase
    # the budget as you go along).
    #
    # instead, we base it on the fact that in lower dimensions, we don't to rely
    # on the prior for too long as the amount of space you need to cover around the
    # prior is fairly low. effectively, since the gp needs little samples to
    # model pretty effectively in low dimension, we can derive the utility from
    # the prior pretty quickly.
    #
    # however, for high dimensional settings, we want to rely longer on the prior
    # for longer as the number of samples needed to model the area around the prior
    # is much larger, and deriving the utility will take longer.
    #
    # in the end, we would like some curve going from 1->0 as n->inf, where `n` is
    # the number of samples we have done so far.
    # the easiest function that does this is `exp(-n)`, with some discounting of `n`
    # dependant on the number of dimensions.
    n_bo_samples = n_sampled_already - initial_design_size
    return math.exp(-n_bo_samples / ndims)


@dataclass
class BayesianOptimization(ScalingOptimizer):
    """Uses `botorch` as an engine for doing bayesian optimiziation."""

    space: SearchSpace | PipelineSpace
    """The search space to use."""

    encoder: ConfigEncoder
    """The encoder to use for encoding and decoding configurations."""

    prior: Prior | None
    """The prior to use for sampling configurations and inferring their likelihood."""

    sample_prior_first: bool
    """Whether to sample the prior configuration first."""

    cost_aware: bool | Literal["log"]
    """Whether to consider the cost of configurations in decision making."""

    n_initial_design: int
    """The number of initial design samples to use before fitting the GP."""

    device: torch.device | None
    """The device to use for the optimization."""

    reference_point: tuple[float, ...] | None = None
    """The reference point to use for the multi-objective optimization."""

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

    def get_constraint_func(self) -> Callable[[Mapping[str, Any]], float] | None:
        assert not isinstance(self.space, PipelineSpace)  # Converted in __post_init__
        # The GP only ever sees the searchables, the cost estimator needs the constants
        return self.budget_constraint(
            self.constraints_func, constants=self.space.constants
        )

    def _cost_estimator_with_constants(self) -> Callable[..., float] | None:
        if self.cost_estimator is None:
            return None
        assert not isinstance(self.space, PipelineSpace)  # Converted in __post_init__
        estimator, constants = self.cost_estimator, self.space.constants
        return lambda **config: estimator(**{**constants, **config})

    def __post_init__(self) -> None:
        if isinstance(self.space, PipelineSpace):
            converted_space = convert_autoscan_to_classic_search_space(self.space)
            if converted_space is not None:
                self.space = converted_space
            else:
                raise ValueError(
                    "This optimizer only supports HPO search spaces, please use an"
                    " AutoScAn"
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

    def sample_candidates(  # noqa: C901, PLR0912, PLR0915  # noqa: C901, PLR0912
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
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
        constraints_func = self.get_constraint_func()

        # Sampled configs are always run at the max fidelity, so that is also what
        # the constraint has to be checked against.
        fidelity_upper = (
            {key: value.upper for key, value in self.space.fidelities.items()}
            if self.space.fidelity is not None
            else {}
        )
        if constraints_func is not None and fidelity_upper:
            _constraints_func = constraints_func
            constraints_func = lambda config: _constraints_func(
                {**config, **fidelity_upper}
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
                    sampler=self.prior if self.prior is not None else "uniform",
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

        # If we should use the prior, weight the acquisition function by
        # the probability of it being sampled from the prior.
        pibo_exp_term = None
        prior = None
        if self.prior:
            pibo_exp_term = _pibo_exp_term(
                len(trials), encoder.ndim, self.n_initial_design
            )
            # If the exp term is insignificant, skip prior acq. weighting
            prior = None if pibo_exp_term < 1e-4 else self.prior

        n_to_acquire = n_to_sample - len(sampled_configs)

        num_objectives = None
        for trial in trials.values():
            if trial.report is not None:
                match trial.report.objective_to_minimize:
                    case None:
                        continue
                    case Sequence():
                        if num_objectives is None:
                            num_objectives = len(trial.report.objective_to_minimize)
                        assert (
                            len(trial.report.objective_to_minimize) == num_objectives
                        ), "All trials must have the same number of objectives."
                    case float():
                        if num_objectives is None:
                            num_objectives = 1
                        assert num_objectives == 1, (
                            "All trials must have the same number of objectives."
                        )
                    case _:
                        raise TypeError(
                            "Objective to minimize must be a float or a sequence "
                            "of floats."
                        )

        # Sanity check, but this shouldn't happen in the first place since we
        # check `n_evaluated < self.n_initial_design` above.
        assert num_objectives is not None, (
            "Either no trials have been completed or"
            " No trials reports have objective values."
        )

        if num_objectives > 1:
            gp = make_default_single_obj_gp(
                x=data.x,
                y=data.y,
                encoder=encoder,
                y_transform=Standardize(m=data.y.shape[-1]),
            )
            if self.reference_point is not None:
                assert len(self.reference_point) == num_objectives, (
                    "Reference point must have the same number of objectives as the"
                    " trials."
                )
                ref_point = torch.tensor(
                    self.reference_point,
                    dtype=data.x.dtype,
                    device=data.x.device,
                )
            else:
                ref_point = torch.tensor(
                    _get_reference_point(data.y.numpy()),
                    dtype=data.x.dtype,
                    device=data.x.device,
                )
            acquisition = qLogNoisyExpectedHypervolumeImprovement(
                model=gp,
                ref_point=ref_point,
                X_baseline=data.x,
                X_pending=data.x_pending,
                prune_baseline=True,
            )
        else:
            if cost_estimator is None:
                gp = make_default_single_obj_gp(x=data.x, y=data.y, encoder=encoder)
            else:
                # Fit a power law to determine GP mean prior
                from autoscan.utils.power_law import fit_power_law_mean_params

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

                gp = make_default_single_obj_gp(
                    x=data.x,
                    y=data.y,
                    encoder=encoder,
                    cost_estimator=cost_estimator,
                    min_cost=mean_params.min_flops,
                    mean_slope=mean_params.slope,
                    mean_bias=mean_params.intercept,
                )
            from botorch.fit import fit_gpytorch_mll
            from gpytorch import ExactMarginalLogLikelihood

            fit_gpytorch_mll(
                ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
            )

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
                bounds = torch.tensor([lower, upper], dtype=torch.float64)

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
                bounds = torch.tensor([lower, upper], dtype=torch.float64)

                candidate_set = (
                    draw_sobol_samples(bounds=bounds, n=512, q=1)
                    .squeeze(1)
                    .to(dtype=torch.float64)
                )
                candidate_set = _feasible_candidates(
                    candidate_set, encoded_constraints_func
                )

                acquisition = qLowerBoundMaxValueEntropy(
                    model=gp,
                    candidate_set=candidate_set,
                    maximize=False,
                )
            elif self.acquisition_func_type == "IT-SCAN-JES":
                import math

                from botorch.fit import fit_gpytorch_mll as _fit_mll
                from gpytorch import ExactMarginalLogLikelihood as _ExactMLL

                from autoscan.optimizers.acquisition.information_theoretic import (
                    CostWeightedModifiedJES,
                )
                from autoscan.optimizers.models.gp import (
                    make_default_single_obj_gp as _make_gp,
                )

                lower = [domain.lower for domain in encoder.domains]
                upper = [domain.upper for domain in encoder.domains]
                bounds = torch.tensor([lower, upper], dtype=torch.float64)

                # Build a candidate set restricted to configs within 3x of C_target.
                raw_candidates = (
                    draw_sobol_samples(bounds=bounds, n=2048, q=1)
                    .squeeze(1)
                    .to(dtype=torch.float64)
                )
                raw_candidates = _feasible_candidates(
                    raw_candidates, encoded_constraints_func
                )
                target_flops = self.target_flops
                assert target_flops is not None
                assert cost_estimator is not None
                flop_vals: list[float] = []
                for cand in raw_candidates:
                    try:
                        conf = encoder.decode(cand.unsqueeze(0))[0]
                        flop_vals.append(float(cost_estimator(**conf)))
                    except Exception:  # noqa: BLE001
                        flop_vals.append(0.0)  # to keep dim consistent
                log_flops = (
                    torch.tensor(flop_vals, dtype=torch.float64).clamp(min=1).log()
                )
                log_target = math.log(target_flops)
                mask = (log_flops - log_target).abs() < math.log(
                    1.5
                )  # a bound of 2/3 to 1.5x the target FLOPs
                candidate_set = (
                    raw_candidates[mask] if mask.sum() >= 4 else raw_candidates
                )

                # Sample S posterior paths over the C_target-filtered candidate set
                # and take the argmin of each path to get fantasy optima at C_target.
                optimal_inputs, optimal_outputs = _sample_optima(
                    gp, candidate_set, num_optima=4
                )

                if (
                    torch.isnan(optimal_inputs).any()
                    or torch.isnan(optimal_outputs).any()
                ):
                    raise ValueError("NaN in modified-JES optimal samples")

                # Build cost GP trained on log-FLOPs for the normalized cost term.
                cost_gp = None
                c_min_log = math.log(1e10)
                c_max_log = math.log(1e19)
                if data.cost is not None and data.cost.numel() >= 2:
                    costs_log = data.cost.clamp(min=1).log()
                    c_min_log = costs_log.min().item()
                    c_max_log = costs_log.max().item()
                    if c_max_log > c_min_log:
                        cost_gp = _make_gp(
                            x=data.x, y=costs_log.unsqueeze(-1), encoder=encoder
                        )
                        _fit_mll(_ExactMLL(likelihood=cost_gp.likelihood, model=cost_gp))

                acquisition = CostWeightedModifiedJES(
                    model=gp,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    cost_model=cost_gp,
                    c_min_log=c_min_log,
                    c_max_log=c_max_log,
                )
            else:
                raise NotImplementedError(
                    "Acquisition function not supported for single objective "
                    "optimization."
                )

        # Prepare acq_options with constraints if provided
        acq_opts = {}
        if encoded_constraints_func is not None:
            acq_opts["nonlinear_inequality_constraints"] = [
                (encoded_constraints_func, True)
            ]

        logger.info("Start Gaussian Process fitting and acquisition...")
        candidates = fit_and_acquire_from_gp(
            gp=gp,
            x_train=data.x,
            encoder=encoder,
            acquisition=acquisition,
            prior=prior,
            n_candidates_required=n_to_acquire,
            pibo_exp_term=pibo_exp_term,
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


def _get_reference_point(loss_vals: np.ndarray) -> np.ndarray:
    """Get the reference point from the completed Trials."""
    eps = 1e-4
    worst_point = np.max(loss_vals, axis=0)
    return worst_point * (1 + eps)
