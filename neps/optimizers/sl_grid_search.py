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
from neps.sampling.priors import Prior
import matplotlib.pyplot as plt
from math import ceil
from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.utils.util import get_trial_config_unique_key
from neps.space.neps_spaces.parameters import PipelineSpace
from neps.optimizers.utils.grid import make_grid
from neps.space.neps_spaces.neps_space import SamplingResolver, SamplingResolutionContext
from neps.space.neps_spaces.neps_space import NepsCompatConverter


if TYPE_CHECKING:
    from neps.space import SearchSpace
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict


logger = logging.getLogger(__name__)

class SL_Grid_Search:
    """The Multi objective algorithm for search space including architectural choices."""


    def __init__(self, space: PipelineSpace, 
                 flops_estimator: Callable[..., int],
                 params_estimator: Callable[..., int],
                 seen_datapoints_estimator: Callable[..., int],
                 max_evaluation_flops: int,
                 ) -> None:
        """Initialize the grid search optimizer.

        Args:
            space: The search space to use, without the fidelity.
        """
        self.space = space
        self.flops_estimator = flops_estimator
        self.params_estimator = params_estimator
        self.seen_datapoints_estimator = seen_datapoints_estimator

        # generate all possible configurations from the pipeline space
        self.config_list = make_grid(
            space,
            ignore_fidelity=True,
            size_per_numerical_hp=10,
        )
        self.down_scaled_conf_list = self.get_downscaled_search_space(max_evaluation_flops=max_evaluation_flops)

        self.none_evaluated_configs: TypeSequence[str] | None = None
        """List of unique keys of configurations that have not been evaluated yet."""
        

    def __call__(  # noqa: C901, PLR0912
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "not supported yet"
        if self.none_evaluated_configs is None:
            self.none_evaluated_configs = [get_trial_config_unique_key(conf) for conf in self.configs_list]
            print("at first call, none evaluated configs:", len(self.none_evaluated_configs))

        nxt_id = len(trials)
        
        config = self.down_scaled_conf_list[nxt_id]
        config_id = str(nxt_id)
        return SampledConfig(config=config, id=config_id, previous_config_id=None)

    # find the specific cut in space for running scaling law
    def get_downscaled_search_space(self, max_evaluation_flops: int) -> None:
        # filter the search pipeline space to only include configurations with flops <= max_evaluation_flops
        confs = []
        for config_dict in self.config_list:
            converted_dict = NepsCompatConverter.from_neps_config(config_dict)
            if self.flops_estimator(**converted_dict) <= max_evaluation_flops:
                confs.append(config_dict)
        return confs

    def extrapolate(self, trials: Mapping[str, Trial], max_target_flop: int) -> dict[str, Any]:
        # considering estimating the flops and number of optimizable parameters is cheap
        # find the closes 
        # fit the trials to a scaling law and extrapolate to target_flop_range
        C, alpha = self.get_power_law_curvature(trials)
        max_n_params = C * (max_target_flop ** alpha)
        
        # find closes config in config_list with n_params less but closest to max_n_params
        conf_list = copy.deepcopy(self.config_list)
        best_param = min([self.params_estimator(**NepsCompatConverter.from_neps_config(conf)) for conf in conf_list if self.params_estimator(**NepsCompatConverter.from_neps_config(conf)) <= max_n_params])
        candidates = [(conf, abs(6 * best_param * self.seen_datapoints_estimator(**NepsCompatConverter.from_neps_config(conf)) - max_target_flop)) for conf in conf_list if self.params_estimator(**NepsCompatConverter.from_neps_config(conf)) == best_param]
        return candidates.sort(key=lambda x: -x[1])[0][0]
    

    def get_power_law_curvature(self, trials: Mapping[str, Trial]) -> tuple[float, float]:
        # assume power law scaling: N_opt = C * (FLOPs)^alpha
        # find C and alpha from existing trials
        flops_list = []
        n_opt_list = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            try:
                flops = self.flops_estimator(**trial.config)
                n_opt = self.params_estimator(**trial.config)
            except Exception:
                logger.error(f"Could not compute FLOPs or N_opt for trial {trial.id}, skipping.")
                continue
            flops_list.append(flops)
            n_opt_list.append(n_opt)
        if len(flops_list) < 2:
            raise ValueError("Not enough data points to fit scaling law.")
        
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
        self.plot_flops_per_objective(trials)
        self.plot_flop_param_ratio(trials)
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
                flops = self.get_total_flops(**trial.config)
            except Exception:
                logger.error(f"Could not compute FLOPs for trial {trial.id}, skipping plot point.")
                continue
            rows.append((trial.id, obj, flops))

        if not rows:
            print("No evaluated trials with objectives/flops to plot.")
            return

        # detect number of objective dims
        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplemented("Multi-objective Scaling law not implemented yet.")

        # x: FLOPs, y: objective
        xs = [r[2] for r in rows]
        ys = [float(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys, marker='o', alpha=0.9)
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs FLOPs")
        ax.grid(True, linestyle="--", alpha=0.4)
        # enforce FLOPs scale to 1e15 on x-axis
        ax.set_xlim(0, 1e15)
        plt.tight_layout()
        plt.savefig("flops_per_objective.png")
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
                flops = self.get_total_flops(**trial.config)
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
        ax.set_xlim(0, 1e15)
        plt.tight_layout()
        plt.savefig("accumulated_flops_per_objective.png")
        return


    def plot_flop_param_ratio(self, trials):
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
                flops = self.get_total_flops(**trial.config)
                n_params = self.get_number_of_parameters(**trial.config)
            except Exception:
                continue
            if n_params == 0:
                continue
            ratio = flops / n_params
            rows.append((trial.id, trial.report.objective_to_minimize, ratio))

        if not rows:
            print("No evaluated trials with objectives/flops/params to plot.")
            return

        first_obj = rows[0][1]
        # If multi-objective, create one subplot per objective dimension where
        # x = FLOPs/Params ratio and y = objective[dim]. For scalar, just plot
        # ratio on x and objective on y.
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplemented("Multi-objective Scaling law not implemented yet.")
        # scalar objective
        xs = [r[2] for r in rows]
        ys = [float(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys, marker='o', alpha=0.9)
        ax.set_xlabel("FLOPs / Params")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs FLOPs / Params")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig("flops_param_ratio.png")
    
    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        n_trials = len(trials)
        imported_configs = []
        imported_keys = []
        for i, (config, result) in enumerate(external_evaluations):
            imported_keys.append(get_trial_config_unique_key(config=config))
            config_id = str(n_trials + i)
            imported_configs.append(
                ImportedConfig(
                    config=config,
                    id=config_id,
                    result=result,
                )
            )
        self.none_evaluated_configs = [
            key for key in self.none_evaluated_configs if key not in imported_keys
        ]
        return imported_configs

