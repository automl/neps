import abc
from logging import getLogger
from typing import Callable, TYPE_CHECKING, Mapping, Any
from neps.optimizers.optimizer import AskFunction

import matplotlib.pyplot as plt

from dataclasses import dataclass
from neps.space import SearchSpace
from neps.state import BudgetInfo, Trial
from neps.optimizers.optimizer import SampledConfig
if TYPE_CHECKING:
    pass

logger = getLogger(__name__)
@dataclass
class ScalingLawGuidedOptimizer:
    space: SearchSpace
    base_optimizer: AskFunction
    max_evaluation_flops: float
    max_target_flops: float
    flops_estimator: Callable
    metric_functions: Mapping[str, Callable]
        

    def __call__(
            self, trials: Mapping[str, Trial], 
            budget_info: BudgetInfo | None = None, n: int | None = None,
        ) -> SampledConfig | list[SampledConfig]:
        return self.base_optimizer(
            trials=trials,
            budget_info=budget_info,
            n=n,
        )

    def callback_on_trial_complete(
        self,
        trials: Mapping[str, Trial],
    ) -> None:
        self.plot_extrapolations(trials=trials)
        self.plot_scaling_laws(trials=trials)
        
    @abc.abstractmethod
    def extrapolate(self, trials: Mapping[str, Trial], max_target_flops: int) -> tuple[dict[str, Any], float]:
        """Extrapolate the performance of a trial to the target flops."""
        pass

    @abc.abstractmethod
    def adapt_search_space(self, trials: Mapping[str, Trial], max_evaluation_flops: int) -> None:
        """Tailor the pipeline based on scaling laws."""
        pass
    

    def plot_scaling_laws(self, trials: Mapping[str, Trial], metric_names: list[str]):
        """Plot scaling laws based on the trials."""
        trials = {
            tid: trial for tid, trial in trials.items()
            if trial.report is not None and trial.report.objective_to_minimize is not None
        }
        metrics = {name: self.metric_functions[name] for name in metric_names if name in self.metric_functions}
        figs, flops_list, objective_values, times = [], [], [], []
        for trial in trials.values():
            times.append(trial.metadata.time_sampled)
            objective_values.append(trial.report.objective_to_minimize)
            try:
                flops_list.append(self.flops_estimator(**trial.config))
            except Exception:
                logger.debug(f"Skipping flops for trial {trial.id}")
                continue
        
        for metric_name, metric_fn in metrics.items():
            metric_values = []
            for trial in trials.values():
                try:
                    metric_values.append(metric_fn(*trial.config))
                except Exception:
                    # metric function may expect different args; skip problematic trials
                    logger.debug(f"Skipping metric for trial {trial.id}")
                    continue

            # two figures: metric vs flops, metric vs objective
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.scatter(flops_list, metric_values, alpha=0.7)
            ax1.set_xlabel("FLOPs")
            ax1.set_ylabel(metric_name)
            ax1.set_title(f"{metric_name} vs FLOPs")
            ax1.grid(True, linestyle="--", alpha=0.4)
            figs.append(fig1)

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(objective_values, metric_values, alpha=0.7)
            ax2.set_xlabel("Objective to minimize")
            ax2.set_ylabel(metric_name)
            ax2.set_title(f"{metric_name} vs Objective")
            ax2.grid(True, linestyle="--", alpha=0.4)
            figs.append(fig2)
        return figs
    
    def plot_extrapolations(self, trials: Mapping[str, Trial], root_dir: str):
        """Plot extrapolations to the target flops."""
        # Implementation of extrapolation plotting logic goes here
        conf, pred = self.extrapolate(trials, self.max_target_flops)
        if not pred:
            return
        trials = {
            tid: trial for tid, trial in trials.items()
            if trial.report is not None and trial.report.objective_to_minimize is not None
        }
        rows = []
        for trial in trials.values():
            rows.append(
                (
                    trial.id, self.flops_estimator(**trial.config), 
                    trial.report.objective_to_minimize, 
                    trial.metadata.time_sampled,
                ),
            )

        xs = [r[1] for r in rows]
        ys = [float(r[2]) for r in rows]
        times = [r[3] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(xs, ys, c=times, cmap='inferno', marker='o', alpha=0.9)
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs FLOPs")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="Time")
        target_flops = self.flops_estimator(**conf)
        ax.scatter(target_flops, pred, c='red', marker='x', s=100, label='Extrapolated Point')
        ax.legend()
        fig.savefig(f"{root_dir}/extraploation.png")
        plt.close(fig)
        return 
