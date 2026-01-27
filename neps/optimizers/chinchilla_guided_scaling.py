from __future__ import annotations

import copy
import itertools
from collections.abc import (
    Mapping,
    Sequence,
    Callable
)

import logging
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
from scipy.optimize import minimize


import matplotlib.pyplot as plt
from neps.optimizers.optimizer import SampledConfig
from neps.optimizers.utils.grid import make_grid
from neps.optimizers.grid_search import GridSearch
from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer

if TYPE_CHECKING:
    from neps.space import SearchSpace
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict


logger = logging.getLogger(__name__)

class Chinchilla_Guided_Scaling(ScalingLawGuidedOptimizer):
    """The Multi objective algorithm for search space including architectural choices."""

    PARAM_ESTIMATOR_KEY = "params_estimator"
    SEEN_DATAPOINTS_ESTIMATOR_KEY = "seen_datapoints_estimator"

    def __init__(self, space: SearchSpace,
                 flops_estimator: Callable[..., int],
                 params_estimator: Callable[..., int],
                 seen_datapoints_estimator: Callable[..., int],
                 max_evaluation_flops: int,
                 max_target_flop: int
                 ) -> None:
        """Initialize the grid search optimizer.

        Args:
            space: The search space to use, without the fidelity.
        """
        self.space = space
        self.flops_estimator = flops_estimator
        self.params_estimator = params_estimator
        self.seen_datapoints_estimator = seen_datapoints_estimator
        self.max_target_flop = max_target_flop
        self.config_list = make_grid(
            space,
            ignore_fidelity=True,
            size_per_numerical_hp=10,
        )
        self.adapt_search_space(trials=None, max_evaluation_cost=max_evaluation_flops)

        super().__init__(
            space=space,
            base_optimizer=GridSearch(
                configs_list=self.config_list
            ),
            max_evaluation_flops=max_evaluation_flops,
            max_target_flops=max_target_flop,
            flops_estimator=flops_estimator,
            metric_functions={
                self.PARAM_ESTIMATOR_KEY: params_estimator, 
                self.SEEN_DATAPOINTS_ESTIMATOR_KEY: seen_datapoints_estimator,
            },
        )

        print(f"Total configurations in search space: {len(self.config_list)}")

    # find the specific cut in space for running scaling law
    def adapt_search_space(
        self,
        trials: Mapping[str, Trial],
        max_evaluation_cost: int,
    ) -> None:
        # filter the search pipeline space to only include configurations with flops <= max_evaluation_flops
        self.config_list.sort(key=lambda conf: self.flops_estimator(**conf))

    def extrapolate(self, trials: Mapping[str, Trial], max_target_flop: int) -> dict[str, Any]:
        # considering estimating the flops and number of optimizable parameters is cheap
        # fit the trials to a scaling law and extrapolate to target_flop_range
        E, A, B, alpha, beta = None, None, None, None, None
        try:
            E, A, B, alpha, beta = self.get_power_law_curvature(trials)
        except Exception as e:
            logger.error(f"Could not extrapolate scaling law: {e}")
            return {}
        
        # find closes config in config_list with n_params less but closest to max_n_params
        conf_list = copy.deepcopy(self.config_list)
        best_candidate = None
        min_loss = float("inf")
        conf_list = [conf for conf in conf_list if self.flops_estimator(**conf) <= max_target_flop]
        for conf in conf_list:
            n, d = self.params_estimator(**conf), self.seen_datapoints_estimator(**conf)
            estimated_loss = E + A / (n ** alpha) + B / (d ** beta)
            if estimated_loss < min_loss:
                min_loss = estimated_loss
                best_candidate = conf
        
        with open("results_chinchilla/scaling_law_extrapolation.txt", "w") as f:
            f.write(f"Find lowest loss for L = {E=} + {A=}/N^{alpha=} {B=}/D^{beta=}\n")
            f.write(f"Selected config for target FLOPs {max_target_flop}: {best_candidate}\n")
            f.write(f"estimated FLOPs: {self.flops_estimator(**best_candidate)}\n")
            f.write(f"estimated N: {self.params_estimator(**best_candidate)}\n")
            f.write(f"estimated D: {self.seen_datapoints_estimator(**best_candidate)}\n")
        return best_candidate

    def get_power_law_curvature(self, trials: Mapping[str, Trial]) -> tuple[float, float, float, float, float]:
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
                N_alpha = np.power(N, alpha, where=(N > 0), out=np.ones_like(N))
                D_beta = np.power(D, beta, where=(D > 0), out=np.ones_like(D))
                
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
            try:
                n_list.append(self.params_estimator(**trial.config))
                d_list.append(self.seen_datapoints_estimator(**trial.config))
                l_list.append(float(trial.report.objective_to_minimize))
            except Exception:
                logger.error(f"Skipping trial {trial.id}")
                continue

        if len(l_list) < 5:
            raise ValueError("Not enough data points to fit scaling law.")

        N = np.array(n_list)
        D = np.array(d_list)
        L = np.array(l_list)

        # Grid of initial guesses
        initialization = list(itertools.product(
            [-1, -0.5, 0, 0.5, 1],         # E
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
                # a, b: [-10, 10] to prevent exp overflow
                # alpha, beta: [0.01, 3] for reasonable power law exponents
                bounds = [
                    (None, None),      # E
                    (-10, 10),         # a
                    (-10, 10),         # b
                    (0.01, 3.0),       # alpha
                    (0.01, 3.0),       # beta
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
                logger.debug(f"Minimize failed for theta0={theta0}: {e}")
                continue

        if best_result is None:
            raise ValueError("Could not fit scaling law with any initialization.")

        E, a, b, alpha, beta = best_result.x
        A = np.exp(np.clip(a, -10, 10))
        B = np.exp(np.clip(b, -10, 10))
        return E, A, B, alpha, beta
        

    def callback_on_trial_complete(
        self,
        trials: Mapping[str, Trial],
    ) -> None:
        """Callback when a trial is completed.

        This is used to update the internal state of the optimizer.

        Args:
            trials: All of the trials that are known about.
        """
        self.extrapolate(trials, max_target_flop=self.max_target_flop)
        self.plot_flops_per_objective(trials)
        self.plot_flop_vs_param(trials)
        self.plot_accumulated_flops_per_objective(trials)
        self.plot_training_curve_envelope(trials)
    
    def plot_training_curve_envelope(self, trials):
        """Plot training curves showing loss vs FLOPs for different parameter counts.
        
        Creates an envelope plot with curves colored by parameter count, similar to
        Chinchilla scaling curves showing how loss improves with FLOPs for fixed model sizes.
        """
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            try:
                flops = self.flops_estimator(**trial.config)
                n_params = self.params_estimator(**trial.config)
                obj = float(trial.report.objective_to_minimize)
            except Exception as e:
                logger.error(f"Could not compute metrics for trial {trial.id}: {e}")
                continue
            if n_params <= 0 or flops <= 0:
                continue
            rows.append((flops, obj, n_params))

        if len(rows) < 2:
            print("Not enough evaluated trials to plot training curve envelope.")
            return

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
            print("No valid parameter groups found.")
            return

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
        plt.savefig("results_chinchilla/training_curve_envelope.png", dpi=150)
        plt.close(fig)
        logger.info("Saved training curve envelope plot to results_chinchilla/training_curve_envelope.png")
    
    def plot_flops_per_objective(self, trials):
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = self.flops_estimator(**trial.config)
            except Exception as e:
                logger.error(f"Could not compute FLOPs for trial {trial.id}, skipping plot point. {e.__str__()}")
                continue
            rows.append((self.params_estimator(**trial.config), obj, flops))

        if not rows:
            print("No evaluated trials with objectives/flops to plot.")
            return

        # detect number of objective dims
        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplemented("Multi-objective Scaling law not implemented yet.")

        # x: FLOPs, y: objective
        times = [r[0] for r in rows]
        xs = [r[2] for r in rows]
        ys = [float(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(xs, ys, c=times, cmap='inferno', marker='o', alpha=0.9)
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs FLOPs")
        ax.grid(True, linestyle="--", alpha=0.4)
        # enforce FLOPs scale to 1e15 on x-axis
        # ax.set_xlim(0, 1e15)
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="Time")
        fig.savefig("results_chinchilla/flops_per_objective.png")
        plt.close(fig)
        return

    def plot_params_vs_loss(self, trials):
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = self.flops_estimator(**trial.config)
            except Exception as e:
                logger.error(f"Could not compute FLOPs for trial {trial.id}, skipping plot point. {e.__str__()}")
                continue
            rows.append((trial.metadata.time_sampled, obj, flops))

        if not rows:
            print("No evaluated trials with objectives/flops to plot.")
            return

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
        ax.set_title("Objective vs FLOPs")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="Time")
        fig.savefig("results_chinchilla/flops_per_objective.png")
        plt.close(fig)
        return

    def plot_accumulated_flops_per_objective(self, trials):
        rows = []
        accumulated_flops = 0
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = self.flops_estimator(**trial.config)
            except Exception:
                logger.error(f"Could not compute FLOPs for trial {trial.id}, skipping plot point.")
                continue
            accumulated_flops += flops
            rows.append((trial.id, obj, accumulated_flops))

        if not rows:
            print("No evaluated trials with objectives/flops to plot.")
            return

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
        plt.savefig("results_chinchilla/accumulated_flops_per_objective.png")
        plt.close(fig)
        return


    def plot_flop_vs_param(self, trials):
        """Plot FLOPs / Params ratio against objectives.

        - Scalar objective: scatter objective vs ratio.
        - Multi-objective: if there are 2 objectives, plot objective0 vs objective1
          and color points by ratio; otherwise fallback to plotting objective[0]
          vs ratio.
        """
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            try:
                flops = self.flops_estimator(**trial.config)
                n_params = self.params_estimator(**trial.config)
            except Exception:
                continue
            if n_params == 0:
                continue
            # ratio = flops / n_params
            rows.append((flops, n_params))

        if not rows:
            print("No evaluated trials with flops/params to plot.")
            return

        xs = [r[0] for r in rows] # FLOPs 
        ys = [float(r[1]) for r in rows] # Params
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(xs, ys, marker='o', alpha=0.9)
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Params")
        ax.set_title("FLOPs vs Params")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig("results_chinchilla/flops_param_ratio.png")
        plt.close(fig)
        return
    
