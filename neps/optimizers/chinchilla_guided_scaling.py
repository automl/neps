from __future__ import annotations

import copy
import itertools
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass

import logging
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, TimeoutError


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from neps.state import BudgetInfo, Trial

from neps.optimizers.optimizer import Artifact, ArtifactType
from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
from neps.optimizers.grid_search import GridSearch
from neps.optimizers.optimizer import ImportedConfig
if TYPE_CHECKING:
    from neps.state import Trial
    from neps.state.pipeline_eval import UserResultDict


logger = logging.getLogger(__name__)


@dataclass
class ScalingLawParameters:
    """Fitted scaling law parameters: L = E + A/N^α + B/D^β"""
    E: float
    A: float
    B: float
    alpha: float
    beta: float

class Chinchilla_Guided_Scaling(ScalingLawGuidedOptimizer):
    """The Multi objective algorithm for search space including architectural choices."""

    def __init__(self,
                space,
                base_optimizer,
                flops_estimator,
                params_estimator,
                seen_datapoints_estimator,
                max_evaluation_flops,
                ) -> None:
        
        self.base_optimizer = base_optimizer
        
        self._scaling_law_params: ScalingLawParameters | None = None
        self._best_candidate: dict[str, Any] | None = None
        self._best_loss: float | None = None

        self.params_estimator = params_estimator
        self.seen_datapoints_estimator = seen_datapoints_estimator
        self.flops_estimator = flops_estimator
        self.space = space
        self.max_evaluation_flops = max_evaluation_flops
    
    def __call__(self, trials, budget_info=None, n=None):
        to_spend = None
        print(f"se")
        if self.flops_estimator is not None and self.max_evaluation_flops is not None:
            to_spend = self.max_evaluation_flops - sum([self.flops_estimator(**trial.config) for trial in trials.values()])
            print(f"BO_Guided_Scaling: to spend {to_spend} FLOPs")
            if to_spend <= 0:
                raise ValueError("No remaining FLOPs budget to spend on evaluation.")
            self.adapt_search_space(trials=trials, max_evaluation_flops=to_spend)
            
        sample = self.base_optimizer(trials, budget_info=BudgetInfo(cost_to_spend=to_spend), n=n)
        return sample
    
    def adapt_search_space(
        self,
        trials: Mapping[str, Trial] | None,
        max_evaluation_flops: int,
    ) -> None:
        """Adapt the search space constraint based on remaining budget."""
        def constraint_func(conf: Mapping[str, Any]) -> float:
            flops = self.flops_estimator(**conf)
            logger.info(f"Evaluating constraint for config: {conf} with estimated FLOPs: {flops}")
            return max_evaluation_flops - flops

        self.base_optimizer.constraints_func = constraint_func


    def extrapolate(self, trials: Mapping[str, Trial], max_target_flops,) -> dict[str, Any] | None:
        return None

        # considering estimating the flops and number of optimizable parameters is cheap
        # fit the trials to a scaling law and extrapolate to target_flop_range
        E, A, B, alpha, beta = None, None, None, None, None
        try:
            # Call get_power_law_curvature with a 10-second timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(Chinchilla_Guided_Scaling.get_power_law_curvature, trials)
                E, A, B, alpha, beta = future.result(timeout=10)
        except TimeoutError:
            logger.error("Scaling law fitting timed out after 10 seconds")
            return None
        except Exception as e:
            logger.error(f"Could not extrapolate scaling law: {e}")
            return None
        
        # Store scaling law parameters for artifact generation and reuse
        self._scaling_law_params = ScalingLawParameters(
            E=E,
            A=A,
            B=B,
            alpha=alpha,
            beta=beta,
        )
        print(f"Fitted scaling law parameters: E={E}, A={A}, B={B}, alpha={alpha}, beta={beta}")
        # if not isinstance(self.base_optimizer, GridSearch):
        #     logger.error("Currently only GridSearch optimizer is supported for extrapolation.")
        #     return None
        # Find the best fit to the scaling law within the constrwhy ained search space
        # TODO: support other optimizers
        conf_list = copy.deepcopy(self.base_optimizer.configs_list)
        best_candidate = None
        min_loss = float("inf")
        conf_list = [
            conf for conf in conf_list if (
                conf.update(self.space.constants), self.flops_estimator(**conf)
                )[1] <= max_target_flops
        ]
        for conf in conf_list:
            conf.update(self.space.constants)
            n, d = self.params_estimator(**conf), self.seen_datapoints_estimator(**conf)
            estimated_loss = E + A / (n ** alpha) + B / (d ** beta)
            if estimated_loss < min_loss:
                min_loss = estimated_loss
                best_candidate = (conf, n, d, estimated_loss)
        
        # for artifact generation
        self._best_candidate = best_candidate
        self._best_loss = min_loss
        
        return best_candidate

    @classmethod
    def _compute_pareto_front(cls, trial_data: list[tuple]) -> list[tuple]:
        # trial_data: (n, d, loss, flops)
        pareto = []
        for i, (n_i, d_i, l_i, f_i) in enumerate(trial_data):
            dominated = False
            for j, (_, _, l_j, f_j) in enumerate(trial_data):
                if i == j:
                    continue
                if (f_j <= f_i and l_j <= l_i) and (f_j < f_i or l_j < l_i):
                    dominated = True
                    break
            if not dominated:
                pareto.append((n_i, d_i, l_i, f_i))
        return pareto

     
    @classmethod
    def get_power_law_curvature(cls, trials: Mapping[str, Trial]) -> tuple[float, float, float, float, float]:
        # L = E + A / N^alpha + B / D^beta
        # where A = e^a and B = e^b

        def huber(residual, delta=1.0):
            abs_r = np.abs(residual)
            return np.where(
                abs_r <= delta,
                0.5 * residual**2,
                delta * (abs_r - 0.5 * delta),
            )

        def objective(theta, N, D, L):
            E, a, b, alpha, beta = theta
            
            # Clip a and b to prevent overflow (exp(20) ≈ 4.85e8)
            a_clipped = np.clip(a, -10, 10)
            b_clipped = np.clip(b, -10, 10)
            A = np.exp(a_clipped)
            B = np.exp(b_clipped)
            
            # Compute with numerical stability
            with np.errstate(divide='ignore', invalid='ignore'):
                N_alpha = np.power(N, alpha, where=(N > 0), out=np.ones_like(N, dtype=np.float64))
                D_beta = np.power(D, beta, where=(D > 0), out=np.ones_like(D, dtype=np.float64))
                
                # Avoid division by zero
                N_alpha = np.where(N_alpha > 0, N_alpha, 1.0)
                D_beta = np.where(D_beta > 0, D_beta, 1.0)
                
                L_hat = E + A / N_alpha + B / D_beta
            
            # Check for NaN or Inf
            if np.any(~np.isfinite(L_hat)):
                return 1e10  # Large penalty for invalid values
            
            residual = L_hat - L
            loss = np.sum(huber(residual))
            
            if not np.isfinite(loss):
                return 1e10
                
            return loss


        l_list, n_list, d_list = [], [], []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                logger.warning(f"Trial {trial.id} has no extra data. Skipping.")
                continue
            try:
                n_list.append(trial.report.extra[cls.N_PARAM_KEY])
                d_list.append(trial.report.extra[cls.N_DATA_KEY])
                l_list.append(float(trial.report.objective_to_minimize))
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Skipping trial {trial.id}: {e}")
                continue

        if len(l_list) < 5:
            raise ValueError("Not enough data points to fit scaling law.")

        N = np.array(n_list)
        D = np.array(d_list)
        L = np.array(l_list)

        # Grid of initial guesses
        initialization = list(itertools.product(
            [-1, -0.5, 0, 0.5, 1],        # E
            [0, 5, 10, 15, 20, 25],       # a
            [0, 5, 10, 15, 20, 25],       # b
            [0, 0.5, 1.0, 1.5, 2.0],      # alpha
            [0, 0.5, 1.0, 1.5, 2.0],      # beta
        ))

        best_result = None
        best_loss = float("inf")

        for theta0 in initialization:

            try:
                # Bounds to prevent numerical issues
                # E: unbounded (can be negative or positive)
                # a, b: [0, 10] to prevent exp overflow
                # alpha, beta: [0.01, 1.0] for reasonable power law exponents
                bounds = [
                    (None, None),      # E
                    (0, 10),         # a
                    (0, 10),         # b
                    (0.01, 1.0),       # alpha
                    (0.01, 1.0),       # beta
                ]
                
                res = minimize(
                    objective,
                    theta0,
                    args=(N, D, L),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_result = res
            except Exception as e:
                raise e

        if best_result is None:
            raise ValueError("Could not fit scaling law with any initialization.")

        E, a, b, alpha, beta = best_result.x
        A = np.exp(np.clip(a, -10, 10))
        B = np.exp(np.clip(b, -10, 10))
        return E, A, B, alpha, beta
    
    def _plot_scaling_trend(self, trials: Mapping[str, Trial], scaling_params: ScalingLawParameters) -> plt.Figure | None:
        """Generate 3D scaling law visualization with Loss vs N vs D.
        
        Shows evaluated trials as points in 3D space and highlights best candidate if available.
        
        Args:
            trials: All evaluated trials.
            scaling_params: ScalingLawParameters object containing fitted parameters.
            
        Returns:
            matplotlib Figure with 3D scatter plot or None if insufficient data.
        """
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                continue
            try:
                n_params = trial.report.extra[self.N_PARAM_KEY] * 1_000_000
                n_data = trial.report.extra[self.N_DATA_KEY]
                loss = float(trial.report.objective_to_minimize)
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}: {e}")
                continue
            if n_params <= 0 or n_data <= 0:
                continue
            rows.append((n_params, n_data, loss))
        
        if not rows:
            logger.warning("No evaluated trials with N/D/loss to plot.")
            return None
        
        # Extract coordinates
        ns = [r[0] for r in rows]
        ds = [r[1] for r in rows]
        losses = [r[2] for r in rows]
        
        # Create 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all evaluated trials
        ax.scatter(ns, ds, losses, c='blue', marker='o', s=50, alpha=0.6, label='Evaluated trials')
        
        # Highlight best candidate if available
        if self._best_candidate is not None:
            try:
                _, best_n, best_d, best_loss = self._best_candidate
                ax.scatter([best_n], [best_d], [best_loss], c='red', marker='*', s=500, 
                          alpha=1.0, edgecolors='black', linewidth=2, label='Best candidate')
            except Exception as e:
                logger.warning(f"Could not plot best candidate: {e}")
        
        # Set axes labels
        ax.set_xlabel('Number of Parameters (N)', fontsize=10)
        ax.set_ylabel('Number of Data Points (D)', fontsize=10)
        ax.set_zlabel('Loss', fontsize=10)
        
        # Use log scale for better visualization
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Title with fitted parameters
        title = f"Scaling Law Fit: L = {scaling_params.E:.4f} + {scaling_params.A:.4e}/N^{scaling_params.alpha:.4f} + {scaling_params.B:.4e}/D^{scaling_params.beta:.4f}"
        ax.set_title(title, fontsize=12, pad=20)
        
        ax.legend(loc='best')
        plt.tight_layout()
        return fig
    
    def get_trial_artifacts(self, trials: Mapping[str, Trial] | None = None) -> list[Artifact] | None:
        """Return scaling law artifacts including parent plots and custom scaling trend.

        Consolidates all artifacts: shared scaling law visualization plots from the base class
        plus Chinchilla-specific scaling trend plot showing fitted scaling law parameters.

        Args:
            trials: Mapping of trial IDs to Trial objects. Required to generate plots.

        Returns:
            List of Artifact objects combining parent class plots and custom scaling trend plot.
        """
        # Get base class artifacts (class method needs to be called on the class)
        artifacts = super().get_trial_artifacts(trials) or []
        
        # Add Chinchilla-specific scaling trend plot if parameters were fitted
        if trials is not None and self._scaling_law_params is not None:
            try:
                fig_scaling = self._plot_scaling_trend(trials, self._scaling_law_params)
                if fig_scaling is not None:
                    artifacts.append(
                        Artifact("scaling_law_trend", fig_scaling, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate scaling law trend plot: {e}")
        
        return artifacts
    
    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        return self.base_optimizer.import_trials(
            external_evaluations=external_evaluations,
            trials=trials,
        )
