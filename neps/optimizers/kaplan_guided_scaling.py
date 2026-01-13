from __future__ import annotations

import copy
from collections.abc import (
    Mapping,
    Sequence,
    Sequence as TypeSequence, Callable
)

import logging
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np



from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
from neps.sampling.priors import Prior
import matplotlib.pyplot as plt
from math import ceil
from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.utils.util import get_trial_config_unique_key
from neps.space.neps_spaces.parameters import PipelineSpace
from neps.optimizers.utils.grid import make_grid
from neps.optimizers.grid_search import GridSearch

if TYPE_CHECKING:
    from neps.space import SearchSpace
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict


logger = logging.getLogger(__name__)

class Kaplan_Guided_Scaling(ScalingLawGuidedOptimizer):
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
        self.adapt_search_space(trials=None, max_evaluation_flops=max_evaluation_flops)

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
    def adapt_search_space(self, trials: Mapping[str, Trial], max_evaluation_flops: int) -> None:
        # filter the search pipeline space to only include configurations with flops <= max_evaluation_flopss
        self.config_list.sort(key=lambda conf: self.flops_estimator(**conf))

    def extrapolate(self, trials: Mapping[str, Trial], max_target_flop: int) -> dict[str, Any]:
        # considering estimating the flops and number of optimizable parameters is cheap
        # fit the trials to a scaling law and extrapolate to target_flop_range
        try:
            C, alpha = self.get_power_law_curvature(trials)
        except Exception as e:
            logger.error(f"Could not extrapolate scaling law: {e}")
            return {}
        max_n_params = C * (max_target_flop ** alpha)
        
        # find closes config in config_list with n_params less but closest to max_n_params
        conf_list = copy.deepcopy(self.config_list)
        n_param_valid_confs = [self.params_estimator(**conf) for conf in conf_list if self.params_estimator(**conf) <= max_n_params]
        if len(n_param_valid_confs) == 0:
            logger.error("No valid configurations found for extrapolated number of parameters.")
            return {}
        best_param = max(n_param_valid_confs)
        candidates = [
            (conf, abs(self.flops_estimator(**conf) - max_target_flop)) 
            for conf in conf_list if self.params_estimator(**conf) == best_param
        ]
        candidates.sort(key=lambda x: x[1])
        best_cadidate = candidates[0][0]
        # write in a text file the C and alpha values and the selected config
        with open("results3/scaling_law_extrapolation.txt", "w") as f:
            f.write(f"C: {C}, alpha: {alpha}\n")
            f.write(f"Find closes point to N_opt: {max_n_params}\n")
            f.write(f"N: {best_param} D: {self.seen_datapoints_estimator(**best_cadidate)} C: {self.flops_estimator(**best_cadidate)}\n")
            f.write(f"Selected config for target FLOPs {max_target_flop}: {best_cadidate}\n")
        return best_cadidate

    def get_power_law_curvature(self, trials: Mapping[str, Trial]) -> tuple[float, float]:
        # assume power law scaling: N_opt = C * (FLOPs)^alpha
        # find C and alpha from existing trials
        flops_list = []
        n_opt_list = []
        param_to_loss_flop = {}
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            try:
                flops = self.flops_estimator(**trial.config)
                n_opt = self.params_estimator(**trial.config)
                objective = trial.report.objective_to_minimize
            except Exception:
                logger.error(f"Could not compute FLOPs or N_opt for trial {trial.id}, skipping.")
                continue
            param_to_loss_flop[n_opt] = (flops, objective) if n_opt not in param_to_loss_flop or objective < param_to_loss_flop[n_opt][1] else param_to_loss_flop[n_opt]
        if len(param_to_loss_flop) < 2:
            raise ValueError("Not enough data points to fit scaling law.")
        
        n_opt_list = list(param_to_loss_flop.keys())
        flops_list = [val[0] for _, val in param_to_loss_flop.items()]
        log_flops = np.log(flops_list)
        log_n_opt = np.log(n_opt_list)
        A = np.vstack([log_flops, np.ones(len(log_flops))]).T
        alpha, log_C = np.linalg.lstsq(A, log_n_opt, rcond=None)[0]
        C = np.exp(log_C)
        # find the N_opt for target_flops from config_list that is closest to the 
        return C, alpha
        

    def callback_on_trial_complete(
        self,
        trials: Mapping[str, Trial],
    ) -> None:
        """Callback when a trial is completed.

        This is used to update the internal state of the optimizer.

        Args:
            trials: All of the trials that are known about.
        """
        # TODO: delicate writing plots and info on disk to runtime
        self.extrapolate(trials, max_target_flop=self.max_target_flop)
        self.plot_flops_per_objective(trials)
        self.plot_flop_vs_param(trials)
        self.plot_accumulated_flops_per_objective(trials)
    
    def plot_flops_per_objective(self, trials):
        """Plot FLOPs vs each objective.

        - For scalar objective: single scatter (objective vs flops).
        - For multi-objective (vector): one subplot per objective dimension.

        The plot is shown with plt.show(). If you want to save, call
        plt.savefig(...) after calling this method.
        """
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
        fig.savefig("results3/flops_per_objective.png")
        plt.close(fig)
        return

    def plot_params_vs_loss(self, trials):
        """Plot FLOPs vs each objective.

        - For scalar objective: single scatter (objective vs flops).
        - For multi-objective (vector): one subplot per objective dimension.

        The plot is shown with plt.show(). If you want to save, call
        plt.savefig(...) after calling this method.
        """
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
        # enforce FLOPs scale to 1e15 on x-axis
        # ax.set_xlim(0, 1e15)
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="Time")
        fig.savefig("results3/flops_per_objective.png")
        plt.close(fig)
        return

    def plot_accumulated_flops_per_objective(self, trials):
        """Plot accumulated FLOPs vs each objective.

        - For scalar objective: single scatter (objective vs accumulated flops).
        - For multi-objective (vector): one subplot per objective dimension.

        The plot is shown with plt.show(). If you want to save, call
        plt.savefig(...) after calling this method.
        """
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
        # enforce FLOPs scale to 1e15 on x-axis
        # ax.set_xlim(0, 1e15)
        plt.tight_layout()
        plt.savefig("results3/accumulated_flops_per_objective.png")
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

        # first_obj = rows[0][1]
        # If multi-objective, create one subplot per objective dimension where
        # x = FLOPs/Params ratio and y = objective[dim]. For scalar, just plot
        # ratio on x and objective on y.
        # if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
        #     raise NotImplemented("Multi-objective Scaling law not implemented yet.")
        # scalar objective
        xs = [r[0] for r in rows] # FLOPs 
        ys = [float(r[1]) for r in rows] # Params
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(xs, ys, marker='o', alpha=0.9)

        try:
            C, alpha = self.get_power_law_curvature(trials)
            # Generate points for the fitted curve
            flops_range = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 100)
            params_fitted = C * (flops_range ** alpha)
            ax.plot(flops_range, params_fitted, 'r--', linewidth=2, label=f'Power-law fit: N = {C:.2e} * F^{alpha:.2f}')
        except Exception as e:
            logger.warning(f"Could not fit power-law curvature: {e}")

        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Params")
        ax.set_title("FLOPs vs Params")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig("results3/flops_param_ratio.png")
        plt.close(fig)
        return
    
