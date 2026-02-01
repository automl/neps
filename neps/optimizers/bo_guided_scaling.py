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


from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
import matplotlib.pyplot as plt
from neps.optimizers.utils.grid import make_grid
from neps.optimizers.bayesian_optimization import BayesianOptimization

from neps.state import Trial, BudgetInfo
if TYPE_CHECKING:
    from neps.space import SearchSpace


logger = logging.getLogger(__name__)

class BO_Guided_Scaling(ScalingLawGuidedOptimizer):
    """The Multi objective algorithm for search space including architectural choices."""

    PARAM_ESTIMATOR_KEY = "params_estimator"
    SEEN_DATAPOINTS_ESTIMATOR_KEY = "seen_datapoints_estimator"

    def __init__(self, space: SearchSpace,
                 bayesian_optimizer: BayesianOptimization,
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
        self.bayesian_optimizer = bayesian_optimizer
        self.max_target_flop = max_target_flop
        self.adapt_search_space(trials=None, max_evaluation_cost=max_evaluation_flops)
        self.bayesian_optimizer.cost_estimator = flops_estimator
        super().__init__(
            space=space,
            base_optimizer=bayesian_optimizer,
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
        print(f"BO_Guided_Scaling: to spend {to_spend} FLOPs")
        self.adapt_search_space(trials=trials, max_evaluation_cost=to_spend)
        sample = self.bayesian_optimizer(trials, budget_info, n)
        self.extrapolate(trials, self.max_target_flop)
        return sample
    
    def get_constraint_func(self) -> Callable[[Mapping[str, Any]], Sequence[float]] | None:
        return self.bayesian_optimizer.constraints_func
    
    def adapt_search_space(
        self,
        trials: Mapping[str, Trial],
        max_evaluation_cost: int,
    ) -> None:
        def constraint_func(conf: Mapping[str, Any]) -> float:
            return max_evaluation_cost - self.flops_estimator(**conf)

        self.bayesian_optimizer.constraints_func = constraint_func

    def extrapolate(self, trials: Mapping[str, Trial], max_target_flop: int) -> tuple[dict[str, Any], float]:
        self.adapt_search_space(trials=trials, max_evaluation_cost=max_target_flop)
        conf, pred = self.bayesian_optimizer.extrapolate(trials, budget_info=BudgetInfo(cost_to_spend=max_target_flop))
        if isinstance(conf, Sequence):
            conf = conf[0]
        return conf, pred
        

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
        flops_list = [val[0] for _, val in param_to_loss_flop.items() ]
        log_flops = np.log(flops_list)
        log_n_opt = np.log(n_opt_list)
        A = np.vstack([log_flops, np.ones(len(log_flops))]).T
        alpha, log_C = np.linalg.lstsq(A, log_n_opt, rcond=None)[0]
        C = np.exp(log_C)
        # find the N_opt for target_flops from config_list that is closest to the 
        return C, alpha
        
