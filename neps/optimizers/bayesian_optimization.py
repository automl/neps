from __future__ import annotations

import logging

import copy
import itertools
import math
from collections.abc import Mapping, Sequence, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from tenacity import Retrying, stop_after_attempt

import matplotlib.pyplot as plt

import numpy as np
import torch
from botorch.acquisition import LinearMCObjective
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.transforms.outcome import Standardize
from botorch.models import SingleTaskGP

from neps.optimizers.models.gp import (
    encode_trials_for_gp,
    fit_and_acquire_from_gp,
    make_default_single_obj_gp,
    encode_constraints_func,
)

from neps.optimizers.optimizer import ImportedConfig, SampledConfig, Artifact, ArtifactType
from neps.optimizers.utils.initial_design import make_initial_design
from neps.space.neps_spaces.neps_space import convert_neps_to_classic_search_space
from neps.space.neps_spaces.parameters import PipelineSpace

if TYPE_CHECKING:
    from neps.sampling import Prior
    from neps.space import ConfigEncoder, SearchSpace
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

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
class BayesianOptimization:
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

    constraints_func: Callable[[Mapping[str, Any]], Sequence[float]] | None = None

    acquisition_func_type: Literal["EI", "IT-JES", "IT-MES"] = "EI" # Expected Improvement, Information Theoretic
    cost_estimator: Callable[..., int] | None = None


    def get_constraint_func(self) -> Callable[[Mapping[str, Any]], Sequence[float]] | None:
        return self.constraints_func
    
    def __post_init__(self) -> None:
        if isinstance(self.space, PipelineSpace):
            converted_space = convert_neps_to_classic_search_space(self.space)
            if converted_space is not None:
                self.space = converted_space
            else:
                raise ValueError(
                    "This optimizer only supports HPO search spaces, please use a NePS"
                    " space-compatible optimizer."
                )
        self.design_samples = None

    def __call__(  # noqa: C901, PLR0912, PLR0915  # noqa: C901, PLR0912
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        return self.sample_candidates(trials, budget_info, n=None)[0]

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
            if hasattr(self.encoder, 'numeric_dims') and len(self.encoder.numeric_dims) >= 2:
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
                        f"NaN values detected in extrapolation grid ({nan_mask.sum().item()} values). "
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

        except Exception as e:
            logger.error(f"Failed to generate extrapolation landscape: {e}")
            return None

    def extrapolate(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
    ) -> tuple[Mapping[str, Any], float]:
        """
        Fits the GP to existing trials, finds the global optimum via Thompson Sampling,
        checks constraints, and visualizes the landscape.
        """
        # 1. Get the trained GP model
        # We assume sample_candidates handles the fitting internally
        print("extrapolate called")
        _, gp, _ = self.sample_candidates(trials, budget_info, n=None)
        if gp is None:
            return None, None
        # 2. Generate Optimal Samples (Global Minima of posterior samples)
        from botorch.acquisition.utils import get_optimal_samples
        
        # Ensure bounds are on the correct device/dtype
        lower = [domain.lower for domain in self.encoder.domains]
        upper = [domain.upper for domain in self.encoder.domains]
        bounds = torch.tensor([lower, upper], dtype=torch.float64)
        
        # Get samples (candidates for the global minimum)
        # maximize=False because we are minimizing Loss
        optimal_inputs, optimal_outputs = get_optimal_samples(
            model=gp, 
            bounds=bounds, 
            num_optima=100,
            maximize=False
        )
        
        # 3. Filter Constraints (Find best VALID input)
        best_idx = None
        best_output = float('inf')
        
        # Decode all at once if supported, otherwise loop is fine for 100 items
        decoded_optimal_configs = self.encoder.decode(optimal_inputs)
        
        for i in range(len(decoded_optimal_configs)):
            config = decoded_optimal_configs[i]
            is_valid = True
            
            if self.constraints_func is not None:
                constraints_vals = self.constraints_func(config)
                # constraints >= 0.0 means satisfied
                if constraints_vals < 0:
                    is_valid = False
            
            if is_valid:
                val = optimal_outputs[i].item()
                if val < best_output:
                    best_output = val
                    best_idx = i
        
        # Fallback: If no point satisfied constraints, pick the best unconstrained one
        if best_idx is None:
            print("WARNING: All extrapolated samples violated constraints. Returning best unconstrained.")
            best_idx = torch.argmin(optimal_outputs).item()
            best_output = optimal_outputs[best_idx].item()

        best_input_tensor = optimal_inputs[best_idx]
        best_config = decoded_optimal_configs[best_idx]

        print(f"Proposed optimal input: {best_config}")
        print(f"Predicted optimal output: {best_output}")

        # 4. Generate and cache visualization
        fig = self._plot_extrapolation_landscape(gp, best_input_tensor, bounds)
        if fig is not None:
            if not hasattr(self, '_artifacts'):
                self._artifacts = {}
            self._artifacts['extrapolation'] = fig

        return best_config, best_output

    def sample_candidates(  # noqa: C901, PLR0912, PLR0915  # noqa: C901, PLR0912
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> tuple[SampledConfig | list[SampledConfig], SingleTaskGP, Any]:
        # If fidelities exist, sample from them as normal
        # This is a bit of a hack, as we set them to max fidelity
        # afterwards, but we need the complete space to sample
        if self.space.fidelity is not None:
            parameters = {**self.space.searchables, **self.space.fidelities}
        else:
            parameters = {**self.space.searchables}

        n_to_sample = 1 if n is None else n
        n_sampled = len(trials)
        id_generator = iter(str(i) for i in itertools.count(n_sampled + 1))

        # If the amount of configs evaluated is less than the initial design
        # requirement, keep drawing from initial design
        n_evaluated = sum(
            1
            for trial in trials.values()
            if (trial.report is not None and trial.report.objective_to_minimize is not None) or (trial.metadata.state == "evaluating")
        )
        sampled_configs: list[SampledConfig] = []

        if n_evaluated < self.n_initial_design:
            # For reproducibility, we need to ensure we do the same sample of all
            # configs each time.
            if self.design_samples is not None and len(self.design_samples) >= n_evaluated + n_to_sample:
                print(f"Using cached design samples for initial design: {len(self.design_samples)} samples available.")
            else:
                self.design_samples = make_initial_design(
                    parameters=parameters,
                    encoder=self.encoder,
                    sample_prior_first=self.sample_prior_first if n_sampled == 0 else False,
                    sampler=self.prior if self.prior is not None else "uniform",
                    seed=None,  # TODO: Seeding, however we need to avoid repeating configs
                    sample_size=self.n_initial_design,
                    constraints_func=self.constraints_func,
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
                return sampled_configs[0] if n is None else sampled_configs, None, None

        # Otherwise, we encode trials and setup to fit and acquire from a GP
        data, encoder = encode_trials_for_gp(
            trials,
            parameters,
            device=self.device,
            encoder=self.encoder,
        )
        encoded_constraints_func = encode_constraints_func(
            constraints_func=self.constraints_func,
            encoder=encoder,
            device=self.device,
        )

        cost_percent = None
        if self.cost_aware:
            # TODO: Interaction with `"log"` cost aware
            # if self.cost_aware == "log":
            #     raise NotImplementedError("Log cost aware not implemented yet.")

            if budget_info is None:
                raise ValueError(
                    "Must provide a 'cost' to configurations if using cost"
                    " with BayesianOptimization."
                )
            if budget_info.cost_to_spend is None:
                raise ValueError("Cost budget must be set if using cost")
            cost_percent = budget_info.used_cost_budget / budget_info.cost_to_spend

        # If we should use the prior, weight the acquisition function by
        # the probability of it being sampled from the prior.
        pibo_exp_term = None
        prior = None
        if self.prior:
            pibo_exp_term = _pibo_exp_term(n_sampled, encoder.ndim, self.n_initial_design)
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
            assert self.cost_estimator is not None, "cost_estimator must be provided for single objective optimization."
            gp = make_default_single_obj_gp(
                x=data.x, y=data.y, encoder=encoder,
                # flop_estimator=self.cost_estimator,
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
                bounds = torch.tensor([lower, upper], dtype=torch.float64)
                print(bounds)
                # torch.autograd.set_detect_anomaly(True)
                optimal_inputs, optimal_outputs = get_optimal_samples(model=gp, bounds=bounds, num_optima=4, maximize=False)
                acquisition = qJointEntropySearch(
                    model=gp,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    condition_noiseless=False,
                    estimation_type='LB',
                    X_pending=data.x_pending,
                    maximize=False,
                )
            elif self.acquisition_func_type == "IT-MES":
                from botorch.acquisition.max_value_entropy_search import qMultiFidelityLowerBoundMaxValueEntropy
                from botorch.acquisition.utils import get_optimal_samples
                lower = [domain.lower for domain in encoder.domains]
                upper = [domain.upper for domain in encoder.domains]
                bounds = torch.tensor([lower, upper], dtype=torch.float64)
                optimal_inputs, optimal_outputs = get_optimal_samples(model=gp, bounds=bounds, num_optima=10, maximize=False)
                acquisition = qMultiFidelityLowerBoundMaxValueEntropy(
                    model=gp,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    condition_noiseless=True,
                    estimation_type='LB',
                    X_pending=data.x_pending,
                    maximize=False,
                )
            else:
                raise NotImplemented("Acquisition function not supported for single objective optimization.")
        
        # Prepare acq_options with constraints if provided
        acq_opts = {}
        if self.constraints_func is not None:
            acq_opts["nonlinear_inequality_constraints"] = [(encoded_constraints_func, True)]
        
        candidates = fit_and_acquire_from_gp(
            gp=gp,
            x_train=data.x,
            encoder=encoder,
            acquisition=acquisition,
            prior=prior,
            n_candidates_required=n_to_acquire,
            pibo_exp_term=pibo_exp_term,
            costs=data.cost if self.cost_aware is not False else None,
            cost_percentage_used=cost_percent,
            costs_on_log_scale=self.cost_aware == "log",
            acq_options=acq_opts,
            hide_warnings=True,
        )

        configs = encoder.decode(candidates)
        for config in configs:
            config.update(self.space.constants)
            if self.space.fidelity is not None:
                config.update(
                    {key: value.upper for key, value in self.space.fidelities.items()}
                )

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
        cand = sampled_configs[0] if n is None else sampled_configs

        figs = self.plot_gp_with_candidates(gp, data.x, data.y, candidates=candidates, dim_x=0, dim_y=2, return_figures=True)
        if figs:
            self._artifacts = figs
        else:
            self._artifacts = {}

        with torch.no_grad():
            # .posterior() creates the distribution at this point
            posterior = gp.posterior(candidates)
            
            # Extract the Mean (Expected Value) with NaN robustness
            pred_mean = posterior.mean
            
            # ROBUST NaN HANDLING: Replace any NaN values with the median of training data
            # This prevents NaN propagation to weights and other computations
            if torch.isnan(pred_mean).any():
                # Get median from training data
                y_train = gp.train_targets
                fill_value = torch.nanmedian(y_train).item()
                
                # Replace NaNs with median fill value
                nan_mask = torch.isnan(pred_mean)
                pred_mean[nan_mask] = fill_value
                
                logger.warning(
                    f"NaN values detected in GP posterior ({nan_mask.sum().item()} values). "
                    f"Replaced with training data median ({fill_value:.6f})."
                )
        
        return cand, gp, pred_mean

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        trials_len = len(trials)
        return [
            ImportedConfig(
                id=str(i),
                config=copy.deepcopy(config),
                result=copy.deepcopy(result),
            )
            for i, (config, result) in enumerate(
                external_evaluations, start=trials_len + 1
            )
        ]

    def get_trial_artifacts(self, trials: Mapping[str, Trial] | None = None) -> list[Artifact] | None:
        """Return Bayesian Optimization specific artifacts.

        Returns artifacts from BO-specific plots (GP posterior, candidates, etc).

        Args:
            trials: Mapping of trial IDs to Trial objects. Used for context.

        Returns:
            List of Artifact objects from BO plots, or empty list if no artifacts generated.
        """
        artifacts = []
        
        # If we have cached figures, convert them to artifacts
        if hasattr(self, '_artifacts') and self._artifacts:
            try:
                for fig_name, fig in self._artifacts.items():
                    artifacts.append(
                        Artifact(f"bo_{fig_name}", fig, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to convert BO figures to artifacts: {e}")
        
        return artifacts

    @staticmethod
    def plot_gp_with_candidates(
        gp: SingleTaskGP,
        x: torch.Tensor,
        y: torch.Tensor,
        candidates: torch.Tensor,
        dim_x: int = 0,
        dim_y: int = 1,
        return_figures: bool = True,
    ) -> dict[str, plt.Figure] | None:
        """
        Plots a 2D slice of the GP posterior mean, existing data, and new candidates.
        Also creates 1D slice plots for each dimension showing mean, uncertainty, and observed points.

        Args:
            gp: The trained BoTorch/GPyTorch model.
            x: Training inputs (encoded) [N x D].
            y: Training targets (losses) [N x 1].
            candidates: The new `x` points proposed by the acquisition function [M x D].
            dim_x: Index of the first dimension to plot.
            dim_y: Index of the second dimension to plot.
            return_figures: If True, return figures. Always save to disk.
            
        Returns:
            If return_figures=True: Dict with keys 'posterior' and 'slices' containing Figure objects
            If return_figures=False: None (figures saved to disk)
        """
        
        gp.eval()
        gp.likelihood.eval()

        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        cand_cpu = candidates.detach().cpu()

        x_all = torch.cat([x_cpu[:, dim_x], cand_cpu[:, dim_x]])
        y_all = torch.cat([x_cpu[:, dim_y], cand_cpu[:, dim_y]])

        x_min, x_max = x_all.min().item(), x_all.max().item()
        y_min, y_max = y_all.min().item(), y_all.max().item()

        # Add 20% margin
        margin_x = 0.2 * (x_max - x_min) if (x_max != x_min) else 1.0
        margin_y = 0.2 * (y_max - y_min) if (y_max != y_min) else 1.0

        resolution = 100
        grid_x = torch.linspace(x_min - margin_x, x_max + margin_x, resolution)
        grid_y = torch.linspace(y_min - margin_y, y_max + margin_y, resolution)
        gx, gy = torch.meshgrid(grid_x, grid_y, indexing="xy")

        # 2. Prepare Test Tensor (Slice)
        num_dims = x.shape[1]
        # Create test points on the same device as the GP
        test_x = torch.zeros(resolution * resolution, num_dims, device=x.device)

        # Fill background dimensions with the mean of training data
        # This simulates "holding other parameters constant at average values"
        for i in range(num_dims):
            test_x[:, i] = x[:, i].mean()

        # Overwrite the two active dimensions with our grid
        # We must move grid to the device first
        test_x[:, dim_x] = gx.flatten().to(x.device)
        test_x[:, dim_y] = gy.flatten().to(x.device)

        # 3. Predict with NaN robustness
        with torch.no_grad():
            posterior = gp.posterior(test_x)
            mean = posterior.mean.cpu().view(resolution, resolution)
            
            # ROBUST NaN HANDLING: Replace NaN values in plotting
            if torch.isnan(mean).any():
                y_train = gp.train_targets.cpu()
                fill_value = torch.nanmedian(y_train).item()
                nan_mask = torch.isnan(mean)
                mean[nan_mask] = fill_value
                logger.warning(
                    f"NaN values in plot posterior ({nan_mask.sum().item()} values). "
                    f"Replaced with median ({fill_value:.6f})."
                )

        # 4. Plot 2D Contour
        fig, ax = plt.subplots(figsize=(10, 7))

        # A. Contour (GP Mean)
        # viridis_r: Purple/Dark = Low Loss (Good), Yellow/Light = High Loss (Bad)
        c = ax.contourf(gx.numpy(), gy.numpy(), mean.numpy(), levels=25, cmap='viridis_r', alpha=0.8)
        cbar = plt.colorbar(c)
        cbar.set_label("Predicted Loss (Mean)", rotation=270, labelpad=15)

        # B. Scatter (Existing Training Data)
        ax.scatter(
            x_cpu[:, dim_x],
            x_cpu[:, dim_y],
            c=y_cpu,
            cmap='viridis_r',
            edgecolors='k',
            s=50,
            label="Observed Data"
        )

        # C. Scatter (New Candidates)
        ax.scatter(
            cand_cpu[:, dim_x],
            cand_cpu[:, dim_y],
            c='red',
            marker='*',
            s=250,
            edgecolors='white',
            linewidth=1.5,
            label="New Candidates"
        )

        best_idx = y_cpu.argmin()
        best_x = x_cpu[best_idx]
        
        for i in range(len(cand_cpu)):
                ax.annotate("",
                        xy=(cand_cpu[i, dim_x].item(), cand_cpu[i, dim_y].item()),
                        xytext=(best_x[dim_x].item(), best_x[dim_y].item()),
                        arrowprops=dict(arrowstyle="->", color="white", alpha=0.5, linestyle="--"))

        ax.set_title(f"GP Posterior & Candidates\n(Slice on Dims {dim_x} & {dim_y})")
        ax.set_xlabel(f"Encoded Dim {dim_x}")
        ax.set_ylabel(f"Encoded Dim {dim_y}")
        ax.legend(loc="upper right")

        plt.tight_layout()
        
        # Keep figure reference for returning
        fig_posterior = fig

        # ========================================================================
        # 5. Create 1D slice plots for each dimension
        # ========================================================================
        fig, axes = plt.subplots(num_dims, 1, figsize=(12, 4 * num_dims))
        if num_dims == 1:
            axes = [axes]

        for dim, ax in enumerate(axes):
            # Create 1D grid for this dimension
            dim_min = x_cpu[:, dim].min().item()
            dim_max = x_cpu[:, dim].max().item()
            dim_margin = 0.1 * (dim_max - dim_min) if dim_max != dim_min else 1.0

            test_grid = torch.linspace(dim_min - dim_margin, dim_max + dim_margin, 200)

            # Hold other dims at mean
            test_points_1d = torch.zeros(len(test_grid), num_dims, device=x.device)
            for i in range(num_dims):
                test_points_1d[:, i] = x[:, i].mean()
            test_points_1d[:, dim] = test_grid.to(x.device)

            with torch.no_grad():
                posterior_1d = gp.posterior(test_points_1d)
                mean_1d = posterior_1d.mean.squeeze().cpu().numpy()
                std_1d = posterior_1d.variance.sqrt().squeeze().cpu().numpy()
                
                # ROBUST NaN HANDLING: Replace NaN values in 1D plots
                nan_mask_mean = np.isnan(mean_1d)
                nan_mask_std = np.isnan(std_1d)
                if nan_mask_mean.any() or nan_mask_std.any():
                    y_train_np = gp.train_targets.cpu().numpy()
                    fill_value = np.nanmedian(y_train_np)
                    mean_1d[nan_mask_mean] = fill_value
                    std_1d[nan_mask_std] = 0.0
                    logger.warning(
                        f"NaN in 1D plot dim {dim}: "
                        f"{nan_mask_mean.sum()} mean NaNs, {nan_mask_std.sum()} std NaNs"
                    )

            test_grid_np = test_grid.cpu().numpy()

            ax.plot(test_grid_np, mean_1d, 'b-', linewidth=2, label='GP Mean')
            ax.fill_between(
                test_grid_np,
                mean_1d - 1.96 * std_1d,
                mean_1d + 1.96 * std_1d,
                alpha=0.3,
                color='blue',
                label='95% CI'
            )

            ax.scatter(
                x_cpu[:, dim].numpy(),
                y_cpu.squeeze().numpy(),
                c='green',
                s=60,
                edgecolors='k',
                label='Observed'
            )

            if len(cand_cpu) > 0:
                ax.scatter(
                    cand_cpu[:, dim].numpy(),
                    np.ones(len(cand_cpu)) * y_cpu.min().item(),
                    c='red',
                    marker='*',
                    s=200,
                    label='Candidates'
                )

            ax.set_title(f'Dimension {dim} (others at mean)')
            ax.set_xlabel(f'Dim {dim}')
            ax.set_ylabel('Objective')
            ax.grid(alpha=0.3)

        # One legend for everything
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Keep figure reference for returning
        fig_slices = fig
        
        # Always return figures
        return {
            "posterior": fig_posterior,
            "slices": fig_slices,
        }


def _get_reference_point(loss_vals: np.ndarray) -> np.ndarray:
    """Get the reference point from the completed Trials."""
    eps = 1e-4
    worst_point = np.max(loss_vals, axis=0)
    return worst_point * (1 + eps)
