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
from scipy.optimize import minimize


from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
import matplotlib.pyplot as plt
from neps.optimizers.utils.grid import make_grid
from neps.optimizers.random_search import RandomSearch

from neps.state import Trial, BudgetInfo
if TYPE_CHECKING:
    from neps.space import SearchSpace


logger = logging.getLogger(__name__)

class RandomSearch_Guided_Scaling(ScalingLawGuidedOptimizer):
    """Random search optimizer guided by scaling laws with FLOPs constraints."""

    PARAM_ESTIMATOR_KEY = "params_estimator"
    SEEN_DATAPOINTS_ESTIMATOR_KEY = "seen_datapoints_estimator"

    def __init__(self, space: SearchSpace,
                 random_search: RandomSearch,
                 flops_estimator: Callable[..., int],
                 params_estimator: Callable[..., int],
                 seen_datapoints_estimator: Callable[..., int],
                 max_evaluation_flops: int,
                 max_target_flop: int
                 ) -> None:
        """Initialize the random search guided scaling optimizer.

        Args:
            space: The search space to use, without the fidelity.
        """
        self.space = space
        self.flops_estimator = flops_estimator
        self.params_estimator = params_estimator
        self.seen_datapoints_estimator = seen_datapoints_estimator
        self.random_search = random_search
        self.max_target_flop = max_target_flop
        self.adapt_search_space(trials=None, max_evaluation_cost=max_evaluation_flops)
        self.root_dir = "results_rs"
        super().__init__(
            space=space,
            base_optimizer=random_search,
            max_evaluation_flops=max_evaluation_flops,
            max_target_flops=max_target_flop,
            flops_estimator=flops_estimator,
            metric_functions={
                self.PARAM_ESTIMATOR_KEY: params_estimator, 
                self.SEEN_DATAPOINTS_ESTIMATOR_KEY: seen_datapoints_estimator,
            },
        )


    def __call__(self, trials, budget_info = None, n = None):
        to_spend = budget_info.cost_to_spend - sum([self.flops_estimator(**trial.config) for trial in trials.values()])
        print(f"RandomSearch_Guided_Scaling: to spend {to_spend} FLOPs")
        self.adapt_search_space(trials=trials, max_evaluation_cost=to_spend)
        sample = self.random_search(trials, budget_info, n)
        return sample
    
    def get_constraint_func(self) -> Callable[[Mapping[str, Any]], Sequence[float]] | None:
        return self.random_search.constraints_func
    
    def adapt_search_space(
        self,
        trials: Mapping[str, Trial],
        max_evaluation_cost: int,
    ) -> None:
        def constraint_func(conf: Mapping[str, Any]) -> float:
            return max_evaluation_cost - self.flops_estimator(**conf)

        self.random_search.constraints_func = constraint_func

    def extrapolate(self, trials: Mapping[str, Trial], max_target_flop: int) -> tuple[dict[str, Any], Any]:
        """Extrapolate the best configuration using power law scaling."""
        # Fit the trials to a scaling law and extrapolate to target_flop
        E, A, B, alpha, beta = None, None, None, None, None
        try:
            E, A, B, alpha, beta = self.get_power_law_curvature(trials)
        except Exception as e:
            logger.error(f"Could not extrapolate scaling law: {e}")
            return {}, None
        
        # Find configs within budget and select best based on scaling law
        best_candidate = None
        min_loss = float("inf")
        
        # Sample some candidates and evaluate with scaling law
        try:
            sampled = self.random_search(trials, BudgetInfo(cost_to_spend=max_target_flop), n=10)
            if not isinstance(sampled, list):
                sampled = [sampled]
            
            for sample in sampled:
                conf = sample.config
                try:
                    if self.flops_estimator(**conf) > max_target_flop:
                        continue
                    
                    n = self.params_estimator(**conf)
                    d = self.seen_datapoints_estimator(**conf)
                    estimated_loss = E + A / (n ** alpha) + B / (d ** beta)
                    
                    if estimated_loss < min_loss:
                        min_loss = estimated_loss
                        best_candidate = conf
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Could not sample candidates for extrapolation: {e}")
        
        # Fallback: if no valid config found, sample one randomly
        if best_candidate is None:
            try:
                sample = self.random_search(trials, BudgetInfo(cost_to_spend=max_target_flop), n=1)
                if isinstance(sample, list):
                    best_candidate = sample[0].config
                else:
                    best_candidate = sample.config
            except Exception as e:
                logger.error(f"Failed to extrapolate and no fallback available: {e}")
                return {}, None
        
        try:
            with open(f"{self.root_dir}/scaling_law_extrapolation.txt", "w") as f:
                f.write(f"Find lowest loss for L = {E=} + {A=}/N^{alpha=} + {B=}/D^{beta=}\n")
                f.write(f"Selected config for target FLOPs {max_target_flop}: {best_candidate}\n")
                f.write(f"estimated FLOPs: {self.flops_estimator(**best_candidate)}\n")
                f.write(f"estimated N: {self.params_estimator(**best_candidate)}\n")
                f.write(f"estimated D: {self.seen_datapoints_estimator(**best_candidate)}\n")
        except Exception as e:
            logger.warning(f"Could not write extrapolation results: {e}")
        
        return best_candidate, None

    def get_power_law_curvature(self, trials: Mapping[str, Trial]) -> tuple[float, float, float, float, float]:
        """Fit power law: L = E + A / N^alpha + B / D^beta"""
        
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
            raise ValueError(f"Not enough data points to fit scaling law (need 5, got {len(l_list)}).")

        log_L = np.log(np.array(l_list))
        log_N = np.log(np.array(n_list))
        log_D = np.log(np.array(d_list))

        # Initial guess
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
        # TODO: deligate writing plots and info on disk to runtime
        self.plot_extrapolations(trials=trials, root_dir=self.root_dir)
        # self.plot_scaling_laws(trials=trials, metric_names=[self.PARAM_ESTIMATOR_KEY, self.SEEN_DATAPOINTS_ESTIMATOR_KEY])
