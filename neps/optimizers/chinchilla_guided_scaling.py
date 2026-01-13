from __future__ import annotations

import copy
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

if TYPE_CHECKING:
    from neps.space import SearchSpace
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict


logger = logging.getLogger(__name__)

class Chinchilla_Guided_Scaling:
    """The Multi objective algorithm for search space including architectural choices."""

    PARAM_ESTIMATOR_KEY = "params_estimator"
    SEEN_DATAPOINTS_ESTIMATOR_KEY = "seen_datapoints_estimator"

    def __init__(self, space: SearchSpace,
                 flops_estimator: Callable[..., int],
                 params_estimator: Callable[..., int],
                 seen_datapoints_estimator: Callable[..., int],
                 max_acculumated_evaluation_flops: int,
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
        self.adapt_search_space(trials=None, max_acculumated_evaluation_flops=max_acculumated_evaluation_flops)

        super().__init__(
            space=space,
            base_optimizer=GridSearch(
                configs_list=self.config_list
            ),
            max_acculumated_evaluation_flops=max_acculumated_evaluation_flops,
            max_target_flops=max_target_flop,
            flops_estimator=flops_estimator,
            metric_functions={
                self.PARAM_ESTIMATOR_KEY: params_estimator, 
                self.SEEN_DATAPOINTS_ESTIMATOR_KEY: seen_datapoints_estimator,
            },
        )

        print(f"Total configurations in search space: {len(self.config_list)}")

    # find the specific cut in space for running scaling law
    def adapt_search_space(self, trials: Mapping[str, Trial], max_acculumated_evaluation_flops: int) -> None:
        # filter the search pipeline space to only include configurations with flops <= max_evaluation_flops
        self.config_list.sort(key=lambda conf: self.flops_estimator(**conf))

    def extrapolate(self, trials: Mapping[str, Trial], max_target_flop: int) -> dict[str, Any]:
        # considering estimating the flops and number of optimizable parameters is cheap
        # find the closes 
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
        
        with open("results2/scaling_law_extrapolation.txt", "w") as f:
            f.write(f"Find lowest loss for L = {E=} + {A=}/N^{alpha=} {B=}/D^{beta=}\n")
            f.write(f"Selected config for target FLOPs {max_target_flop}: {best_candidate}\n")
            f.write(f"estimated FLOPs: {self.flops_estimator(**best_candidate)}\n")
            f.write(f"estimated N: {self.params_estimator(**best_candidate)}\n")
            f.write(f"estimated D: {self.seen_datapoints_estimator(**best_candidate)}\n")
        return best_candidate

    def get_power_law_curvature(self, trials: Mapping[str, Trial]) -> tuple[float, float, float, float, float]:
        # L = E + A / N^alpha + B / D^beta
        # optimize in log-space with Huber loss

        def huber(residual, delta=1.0):
            abs_r = np.abs(residual)
            return np.where(
                abs_r <= delta,
                0.5 * residual**2,
                delta * (abs_r - 0.5 * delta),
            )

        def objective(theta, log_N, log_D, log_L):
            E, logA, logB, alpha, beta = theta
            A = np.exp(logA)
            B = np.exp(logB)

            L_hat = E + A * np.exp(-alpha * log_N) + B * np.exp(-beta * log_D)
            log_L_hat = np.log(L_hat)

            return np.sum(huber(log_L_hat - log_L))

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

        log_L = np.log(np.array(l_list))
        log_N = np.log(np.array(n_list))
        log_D = np.log(np.array(d_list))

        # initial guess
        theta0 = np.array([
            np.min(l_list),   # E
            0.0,              # logA
            0.0,              # logB
            0.5,              # alpha
            0.5,              # beta
        ])

        res = minimize(
            objective,
            theta0,
            args=(log_N, log_D, log_L),
            method="L-BFGS-B",
        )

        E, logA, logB, alpha, beta = res.x
        return E, np.exp(logA), np.exp(logB), alpha, beta
        

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
        fig.savefig("results2/flops_per_objective.png")
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
        fig.savefig("results2/flops_per_objective.png")
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
        plt.savefig("results2/accumulated_flops_per_objective.png")
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
        plt.savefig("results2/flops_param_ratio.png")
        plt.close(fig)
        return
    
