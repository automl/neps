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


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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

    FLOPS_KEY = "flops"
    N_PARAM_KEY = "n_param"
    N_DATA_KEY = "n_data"

    def __init__(self,
                space,
                base_optimizer,
                flops_estimator,
                params_estimator,
                seen_datapoints_estimator,
                ) -> None:
        
        self.base_optimizer = base_optimizer
        
        self._scaling_law_params: ScalingLawParameters | None = None
        self._best_candidate: dict[str, Any] | None = None
        self._best_loss: float | None = None

        self.params_estimator = params_estimator
        self.seen_datapoints_estimator = seen_datapoints_estimator
        self.flops_estimator = flops_estimator
        self.space = space

    def extrapolate(self, trials: Mapping[str, Trial], max_target_flops,) -> dict[str, Any] | None:
        # considering estimating the flops and number of optimizable parameters is cheap
        # fit the trials to a scaling law and extrapolate to target_flop_range
        E, A, B, alpha, beta = None, None, None, None, None
        try:
            E, A, B, alpha, beta = Chinchilla_Guided_Scaling.get_power_law_curvature(trials)
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
        if not isinstance(self.base_optimizer, GridSearch):
            logger.error("Currently only GridSearch optimizer is supported for extrapolation.")
            return None
        # Find the best fit to the scaling law within the constrained search space
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
            if trial.report is None or trial.report.objective_to_minimize is None:
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
     
    @classmethod
    def _plot_training_curve_envelope(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate training curves showing loss vs FLOPs for different parameter counts (no disk I/O).
        
        Creates an envelope plot with curves colored by parameter count, similar to
        Chinchilla scaling curves showing how loss improves with FLOPs for fixed model sizes.
        
        Args:
            trials: All evaluated trials.
            
        Returns:
            matplotlib Figure or None if insufficient data.
        """
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            if trial.report.extra is None:
                logger.warning(f"Trial {trial.id} has no extra data. Skipping.")
                continue
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY] * 1_000_000
                obj = float(trial.report.objective_to_minimize)
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}: {e}")
                continue
            if n_params <= 0 or flops <= 0:
                continue
            rows.append((flops, obj, n_params))

        if len(rows) < 2:
            logger.warning("Not enough evaluated trials to plot training curve envelope.")
            return None

        # Sort by parameters
        rows.sort(key=lambda r: r[2])
        
        # Group by parameter count (create bins/levels)
        param_counts = {}
        for flops, obj, n_params in rows:
            # Round to nearest log-scale to group similar parameter counts
            param_bin = round(np.log10(n_params) * 4) / 4  # 4 bins per log decade
            if param_bin not in param_counts:
                param_counts[param_bin] = []
            param_counts[param_bin].append((flops, obj))

        if not param_counts:
            logger.warning("No valid parameter groups found.")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get unique parameter levels for coloring
        param_levels = sorted(param_counts.keys())
        cmap = plt.cm.get_cmap('viridis')
        norm = plt.Normalize(vmin=min(param_levels), vmax=max(param_levels))
        
        # Plot lines for each parameter level
        for param_bin in param_levels:
            points = param_counts[param_bin]
            # Sort by FLOPs for each parameter level
            points.sort(key=lambda p: p[0])
            
            if len(points) >= 2:
                flops_vals = [p[0] for p in points]
                obj_vals = [p[1] for p in points]
                
                # Get parameter count from bin
                param_count = 10 ** param_bin
                color = cmap(norm(param_bin))
                
                ax.plot(flops_vals, obj_vals, marker='o', linestyle='-', 
                       color=color, alpha=0.7, linewidth=2, markersize=4,
                       label=f"{param_count:.2e}" if len(param_levels) <= 10 else "")

        # Formatting
        ax.set_xscale('log')
        ax.set_xlabel('FLOPs', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Curve Envelope: Loss vs FLOPs (colored by Model Size)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Log10(Parameters)', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    @classmethod
    def _plot_flops_per_objective(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate FLOPs vs objective visualization (no disk I/O).
        
        Args:
            trials: All evaluated trials.
            
        Returns:
            matplotlib Figure or None if insufficient data.
        """
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY] * 1_000_000
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}, skipping plot point. {e}")
                continue
            rows.append((n_params, obj, flops))

        if not rows:
            logger.warning("No evaluated trials with objectives/flops to plot.")
            return None

        # detect number of objective dims
        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplemented("Multi-objective Scaling law not implemented yet.")

        # x: FLOPs, y: objective
        n_param = [r[0] for r in rows]
        xs = [r[2] for r in rows]  # natural log of FLOPs
        ys = [float(r[1]) for r in rows]  # natural log of objective
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Normalize color scale based on actual min/max of parameters
        norm = Normalize(vmin=min(n_param), vmax=max(n_param))
        
        sc = ax.scatter(xs, ys, c=n_param, cmap='inferno', marker='o', alpha=0.9, norm=norm)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs FLOPs (log-log scale)")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="Parameters")
        return fig

    @classmethod
    def _plot_pareto_front(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate Pareto front visualization (non-dominated points) in log-log scale.
        
        Plots only the Pareto optimal points where no other point has both
        lower FLOPs and lower objective (loss) in log-log space. Fits a robust linear 
        trend line to the log-scaled Pareto frontier, with higher weight given to 
        points at the end of the curve to better capture the asymptotic behavior.
        
        Args:
            trials: All evaluated trials.
            
        Returns:
            matplotlib Figure or None if insufficient data.
        """
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY] * 1_000_000
                n_data = trial.report.extra[cls.N_DATA_KEY]
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}, skipping plot point. {e}")
                continue
            rows.append((n_params, n_data, float(obj), flops,))

        if not rows:
            logger.warning("No evaluated trials with objectives/flops to plot.")
            return None

        # Compute Pareto front using unified helper function
        pareto_front = cls._compute_pareto_front(rows)

        if not pareto_front:
            logger.warning("No Pareto front points found.")
            return None

        # Sort by FLOPs for better visualization
        pareto_front.sort(key=lambda r: r[3])
        
        # Extract coordinates
        n_params_front = [r[0] for r in pareto_front]
        n_data_front = [r[1] for r in pareto_front]
        objs_front = [r[2] for r in pareto_front]
        flops_front = [r[3] for r in pareto_front]
        
        # Extract all points for context (semi-transparent)
        n_params_all = [r[0] for r in rows]
        objs_all = [r[2] for r in rows]
        flops_all = [r[3] for r in rows]
        
        # Convert to log scale for fitting
        log_flops_front = np.log(flops_front)
        log_objs_front = np.log(objs_front)
        
        # Fit linear trend to log-scaled data with strong emphasis on end of curve
        # Use exponential weights: later points get much higher weight, early points treated as outliers
        n_points = len(log_flops_front)
        # Exponential weighting with higher range (0, 4) treats early FLOPs as outliers
        weights = np.exp(np.linspace(0, 4, n_points))  # Exponential weights strongly favor end points
        weights = weights / np.sum(weights)  # Normalize
        
        try:
            # Fit linear model in log space: log(obj) = slope * log(flops) + intercept
            coeffs = np.polyfit(log_flops_front, log_objs_front, 1, w=weights)
            slope, intercept = coeffs[0], coeffs[1]
            fit_line = np.poly1d(coeffs)
            fitted_log_objs = fit_line(log_flops_front)
            fitted_objs = np.exp(fitted_log_objs)
        except Exception as e:
            logger.warning(f"Could not fit linear trend to Pareto front: {e}")
            slope, intercept, fitted_objs = None, None, None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot all points in light gray (context)
        ax.scatter(flops_all, objs_all, c=n_params_all, cmap='gray', marker='o', 
                  alpha=0.2, s=30, label='All points')
        
        # Plot Pareto front with colors
        norm_front = Normalize(vmin=min(n_params_front), vmax=max(n_params_front))
        sc = ax.scatter(flops_front, objs_front, c=n_params_front, cmap='inferno', 
                       marker='D', s=100, alpha=1.0, edgecolors='black', linewidth=1.5,
                       norm=norm_front, label='Pareto front (actual)')
        
        # Connect Pareto front points with a line
        ax.plot(flops_front, objs_front, 'k-', alpha=0.3, linewidth=1)
        
        # Plot fitted linear trend if available
        if fitted_objs is not None and slope is not None:
            ax.plot(flops_front, fitted_objs, 'r--', alpha=0.7, linewidth=2.5, 
                   label=f'Linear fit (log scale, slope={slope:.4f})')
        
        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel("FLOPs", fontsize=12)
        ax.set_ylabel("Objective to minimize", fontsize=12)
        ax.set_title(f"Pareto Front - Log-Log Scale ({len(pareto_front)}/{len(rows)} points)", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.3, which='both')
        ax.legend(loc='best')
        
        # Add text annotation below the plot with line equation
        if slope is not None and intercept is not None:
            # In log space: log(obj) = slope * log(flops) + intercept
            # Which means: obj = exp(intercept) * flops^slope
            scaler = np.exp(intercept)
            line_eq = f"Log-linear fit: log(y) = {slope:.4f}·log(x) + {intercept:.4f}  →  y = {scaler:.4e}·x^{slope:.4f}"
            fig.text(0.5, 0.02, line_eq, ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.colorbar(sc, ax=ax, label="Parameters")
        return fig

    @classmethod
    def _plot_params_vs_loss(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate params vs loss visualization colored by time (no disk I/O).
        
        Args:
            trials: All evaluated trials.
            
        Returns:
            matplotlib Figure or None if insufficient data.
        """
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}, skipping plot point. {e}")
                continue
            rows.append((trial.metadata.time_sampled, obj, flops))

        if not rows:
            logger.warning("No evaluated trials with objectives/flops to plot.")
            return None

        # detect number of objective dims
        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplemented("Multi-objective Scaling law not implemented yet.")

        # x: FLOPs, y: objective
        times = [r[0] for r in rows]
        xs = [r[2] for r in rows]
        ys = [float(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(xs, ys, c=times, cmap='coolwarm', marker='o', alpha=0.9)
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs FLOPs (colored by Time)")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="Time")
        return fig

    @classmethod
    def _plot_data_vs_loss(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate data points vs loss visualization colored by FLOPs (no disk I/O).
        
        Args:
            trials: All evaluated trials.
            
        Returns:
            matplotlib Figure or None if insufficient data.
        """
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                n_data = trial.report.extra[cls.N_DATA_KEY]
                flops = trial.report.extra[cls.FLOPS_KEY]
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}, skipping plot point. {e}")
                continue
            rows.append((n_data, float(obj), flops))

        if not rows:
            logger.warning("No evaluated trials with data/loss to plot.")
            return None

        # detect number of objective dims
        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplemented("Multi-objective Scaling law not implemented yet.")

        # x: data points, y: objective
        n_data = [r[0] for r in rows]
        xs = [r[0] for r in rows]
        ys = [float(r[1]) for r in rows]
        flops = [r[2] for r in rows]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Normalize color scale based on FLOPs
        norm = Normalize(vmin=min(flops), vmax=max(flops))
        
        sc = ax.scatter(xs, ys, c=flops, cmap='plasma', marker='o', alpha=0.9, norm=norm)
        ax.set_xlabel("Number of Data Points")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs Data Points")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="FLOPs")
        return fig

    @classmethod
    def _plot_accumulated_flops_per_objective(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate accumulated FLOPs vs objective visualization (no disk I/O).
        
        Args:
            trials: All evaluated trials.
            
        Returns:
            matplotlib Figure or None if insufficient data.
        """
        rows = []
        accumulated_flops = 0
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}, skipping plot point. {e}")
                continue
            accumulated_flops += flops
            rows.append((trial.id, obj, accumulated_flops))

        if not rows:
            logger.warning("No evaluated trials with objectives/flops to plot.")
            return None

        # detect number of objective dims
        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplemented("Multi-objective Scaling law not implemented yet.")

        # x: accumulated FLOPs, y: objective
        xs = [r[2] for r in rows]
        ys = [float(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker='o', linestyle='-', alpha=0.9)
        ax.set_xlabel("Accumulated FLOPs")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs Accumulated FLOPs")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        return fig

    @classmethod
    def _plot_flop_vs_param(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate FLOPs vs Params visualization (no disk I/O).

        Args:
            trials: All evaluated trials.
            
        Returns:
            matplotlib Figure or None if insufficient data.
        """
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            if trial.report.extra is None:
                continue
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY] * 1_000_000
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}: {e}")
                continue
            if n_params == 0:
                continue
            rows.append((flops, n_params))

        if not rows:
            logger.warning("No evaluated trials with flops/params to plot.")
            return None

        xs = [r[0] for r in rows]  # FLOPs 
        ys = [float(r[1]) for r in rows]  # Params
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(xs, ys, marker='o', alpha=0.9)
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Params")
        ax.set_title("FLOPs vs Params")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        return fig
    
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
            if trial.report is None or trial.report.objective_to_minimize is None:
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
    
    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        return self.base_optimizer.import_trials(
            external_evaluations=external_evaluations,
            trials=trials,
        )
    
    # def get_trial_artifacts(self, trials: Mapping[str, Trial] | None = None) -> list[Artifact] | None:
    #     artifacts = self.__class__.get_trial_artifacts(trials)
    #     scaling_params = self._scaling_law_params
    #     if scaling_params is not None:
    #         try:
    #             fig_scaling = self._plot_scaling_trend(trials, scaling_params)
    #             if fig_scaling is not None:
    #                 artifacts.append(
    #                     Artifact("scaling_law_trend", fig_scaling, ArtifactType.FIGURE)
    #                 )
    #         except Exception as e:
    #             logger.warning(f"Failed to generate scaling law trend plot: {e}")

    #     return artifacts
    
    @classmethod
    def get_trial_artifacts(cls, trials: Mapping[str, Trial] | None = None) -> list[Artifact] | None:
        """Return scaling law artifacts for runtime persistence.

        Consolidates all artifacts: scaling law visualization plots
        from trials data.

        Args:
            trials: Mapping of trial IDs to Trial objects. Required to generate plots.

        Returns:
            List of Artifact objects, or None if no scaling law has been fitted yet.
        """
        artifacts = []
        
        if trials is not None:            
            try:
                fig_accumulated = cls._plot_accumulated_flops_per_objective(trials)
                if fig_accumulated is not None:
                    artifacts.append(
                        Artifact("accumulated_flops_objective", fig_accumulated, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate accumulated FLOPs plot: {e}")
            
            try:
                fig_flops_params = cls._plot_flop_vs_param(trials)
                if fig_flops_params is not None:
                    artifacts.append(
                        Artifact("flops_vs_params", fig_flops_params, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate FLOPs vs Params plot: {e}")
            
            try:
                fig_flops_obj = cls._plot_flops_per_objective(trials)
                if fig_flops_obj is not None:
                    artifacts.append(
                        Artifact("flops_per_objective", fig_flops_obj, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate FLOPs per objective plot: {e}")
            
            try:
                fig_pareto = cls._plot_pareto_front(trials)
                if fig_pareto is not None:
                    artifacts.append(
                        Artifact("pareto_front", fig_pareto, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate Pareto front plot: {e}")
            
            # Plot: Params vs loss
            try:
                fig_params_loss = cls._plot_params_vs_loss(trials)
                if fig_params_loss is not None:
                    artifacts.append(
                        Artifact("params_vs_loss", fig_params_loss, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate params vs loss plot: {e}")
            
            # Plot: Data vs loss
            try:
                fig_data_loss = cls._plot_data_vs_loss(trials)
                if fig_data_loss is not None:
                    artifacts.append(
                        Artifact("data_vs_loss", fig_data_loss, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate data vs loss plot: {e}")
            
            # Plot: Training curve envelope
            try:
                fig_envelope = cls._plot_training_curve_envelope(trials)
                if fig_envelope is not None:
                    artifacts.append(
                        Artifact("training_curve_envelope", fig_envelope, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate training curve envelope plot: {e}")
        
        return artifacts
