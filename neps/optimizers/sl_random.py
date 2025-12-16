from __future__ import annotations

import copy
from collections.abc import (
    Mapping,
    Sequence,
    Sequence as TypeSequence, Callable
)

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from gpytorch.utils.warnings import NumericalWarning

from neps.optimizers.models.gp import (
    encode_trials_for_gp,
    fit_and_acquire_from_gp,
    make_default_single_obj_gp,
)
from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.sampling.priors import Prior
import matplotlib.pyplot as plt
from math import ceil
from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.utils.util import get_trial_config_unique_key

if TYPE_CHECKING:
    from neps.space import SearchSpace
    from neps.space.encoding import ConfigEncoder
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict


logger = logging.getLogger(__name__)

@dataclass
class SL_PriMO:
    """The Multi objective algorithm for search space including architectural choices."""

    space: SearchSpace
    """The search space to use, without the fidelity."""

    encoder: ConfigEncoder
    """The encoder to use for the search space."""

    initial_design_size: int
    """The number of initial designs to use."""

    # fid_max: int | float
    # """The maximum fidelity value in the BracketOptimizer's search space."""

    # fid_name: str
    # """The name of the fidelity in the BracketOptimizer's search space."""

    get_number_of_parameters: Callable[..., int] # gets the hyperparams selected and spits the num of params
    get_total_flops: Callable[..., int] # gets the hyperparams selected and spits the num of flops for feed forward
    # parameters that have effect on the number of params, then when 
    # we change these we should make sure that the N/D stays in the given range.

    scalarization_weights: dict[str, float] | None = None
    """The scalarization weights to use for the objectives for BO."""

    device: torch.device | None = None
    """The device to use for the GP optimization."""

    priors: Mapping[str, Prior] | None = None
    """The priors to use for this optimizer."""

    n_init_used: int = field(default=0, init=False)
    """The effective number of initial seed configurations used
    for the Bayesian optimization. This refers to the number of
    configurations that were evaluated at the maximum fidelity.
    """

    epsilon: float = 0.25
    """The epsilon value to use for the epsilon-greedy decaying prior-weighted
    acquisition function. This is the probability of not using the prior
    acquisition function.
    """

    config_list: list[dict[str, Any]] = field(default_factory=list)
    """A fixed list of configurations to evaluate in order."""

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
        
        # sampler = Prior.from_parameters(self.space.searchables)
        # configs = sampler.sample(1, to=self.encoder.domains)
        # config_dicts = self.encoder.decode(configs)
        
        # return SampledConfig(id=str(nxt_id), config=config_dicts[0])
        config = self.config_list[nxt_id]
        config_id = str(nxt_id)
        return SampledConfig(config=config, id=config_id, previous_config_id=None)

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
            k = len(first_obj)
        else:
            k = 1

        if k == 1:
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

        # multi-objective: create subplots (one per objective dimension)
        ncols = min(3, k)
        nrows = ceil(k / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for dim in range(k):
            ax = axes[dim]
            xs = [r[2] for r in rows]
            ys = [float(r[1][dim]) for r in rows]
            # connect points with lines and markers
            ax.scatter(xs, ys, marker='o', alpha=0.9)
            ax.set_xlabel("FLOPs")
            ax.set_ylabel(f"Objective[{dim}] to minimize")
            ax.set_title(f"Objective[{dim}] vs FLOPs")
            ax.grid(True, linestyle="--", alpha=0.4)
            # enforce FLOPs scale to 1e15 on x-axis
            ax.set_xlim(0, 1e15)

        # hide any unused axes
        for i in range(k, len(axes)):
            try:
                axes[i].set_visible(False)
            except Exception:
                pass

        plt.tight_layout()
        plt.savefig("flops_per_objective.png")

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
            k = len(first_obj)
        else:
            k = 1

        if k == 1:
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

        # multi-objective: create subplots (one per objective dimension)
        ncols = min(3, k)
        nrows = ceil(k / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for dim in range(k):
            ax = axes[dim]
            xs = [r[2] for r in rows]
            ys = [float(r[1][dim]) for r in rows]
            ax.plot(xs, ys, marker='o', linestyle='-', alpha=0.9)
            ax.set_xlabel("Accumulated FLOPs")
            ax.set_ylabel(f"Objective[{dim}] to minimize")
            ax.set_title(f"Objective[{dim}] vs Accumulated FLOPs")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlim(0, 1e15)
        # hide any unused axes
        for i in range(k, len(axes)):
            try:
                axes[i].set_visible(False)
            except Exception:
                pass
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
            k = len(first_obj)
            ncols = min(3, k)
            nrows = ceil(k / ncols)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

            for dim in range(k):
                ax = axes[dim]
                xs = [r[2] for r in rows]  # ratio on x-axis
                ys = [float(r[1][dim]) for r in rows]
                ax.scatter(xs, ys, marker='o', alpha=0.9)
                ax.set_xlabel("FLOPs / Params")
                ax.set_ylabel(f"Objective[{dim}] to minimize")
                ax.set_title(f"Objective[{dim}] vs FLOPs/Params")
                ax.grid(True, linestyle="--", alpha=0.4)

            # hide any unused axes
            for i in range(k, len(axes)):
                try:
                    axes[i].set_visible(False)
                except Exception:
                    pass

            plt.tight_layout()
            plt.savefig("flops_param_ratio.png")
            return

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

