from __future__ import annotations

import copy
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
from neps.optimizers.optimizer import Artifact, ArtifactType
from neps.optimizers.bayesian_optimization import BayesianOptimization

from neps.state import Trial, BudgetInfo
if TYPE_CHECKING:
    from neps.space import SearchSpace


logger = logging.getLogger(__name__)


@dataclass
class ScalingLawParameters:
    """Fitted scaling law parameters: L = E + A/N^α + B/D^β"""
    E: float
    A: float
    B: float
    alpha: float
    beta: float


class BO_Guided_Scaling(ScalingLawGuidedOptimizer):
    """The Multi objective algorithm for search space including architectural choices."""

    def __init__(self,
                 space: SearchSpace,
                 bayesian_optimizer: BayesianOptimization,
                 flops_estimator: Callable[..., int],
                 params_estimator: Callable[..., int],
                 seen_datapoints_estimator: Callable[..., int],
                 max_evaluation_flops: int,
                 max_target_flops: int
                 ) -> None:
        """Initialize the Bayesian Optimization guided scaling optimizer.

        Args:
            space: The search space to use, without the fidelity.
            bayesian_optimizer: The Bayesian Optimization optimizer instance.
            flops_estimator: Function to estimate FLOPs for a configuration.
            params_estimator: Function to estimate parameters for a configuration.
            seen_datapoints_estimator: Function to estimate data points for a configuration.
            max_evaluation_flops: Maximum FLOPs allowed for evaluation.
            max_target_flop: Target maximum FLOPs for extrapolation.
        """
        self.space = space
        self.flops_estimator = flops_estimator
        self.params_estimator = params_estimator
        self.seen_datapoints_estimator = seen_datapoints_estimator
        self.bayesian_optimizer = bayesian_optimizer
        self.max_target_flops = max_target_flops
        
        self._scaling_law_params: ScalingLawParameters | None = None
        self._best_candidate: dict[str, Any] | None = None
        self._best_loss: float | None = None
        
        self.adapt_search_space(trials=None, max_evaluation_flops=max_evaluation_flops)
        self.bayesian_optimizer.cost_estimator = flops_estimator
        
        super().__init__(
            space=space,
            base_optimizer=bayesian_optimizer,
            max_evaluation_flops=max_evaluation_flops,
            max_target_flops=max_target_flops,
            flops_estimator=flops_estimator,
            metric_functions={
                "params_estimator": params_estimator, 
                "seen_datapoints_estimator": seen_datapoints_estimator,
            },
        )

    def __call__(self, trials, budget_info=None, n=None):
        to_spend = budget_info.cost_to_spend - sum([self.flops_estimator(**trial.config) for trial in trials.values()])
        logger.info(f"BO_Guided_Scaling: to spend {to_spend} FLOPs")
        self.adapt_search_space(trials=trials, max_evaluation_flops=to_spend)
        sample = self.bayesian_optimizer(trials, budget_info, n)
        # self.extrapolate(trials, self.max_target_flops)
        return sample
    
    def adapt_search_space(
        self,
        trials: Mapping[str, Trial] | None,
        max_evaluation_flops: int,
    ) -> None:
        """Adapt the search space constraint based on remaining budget."""
        def constraint_func(conf: Mapping[str, Any]) -> float:
            return max_evaluation_flops - self.flops_estimator(**conf)

        self.bayesian_optimizer.constraints_func = constraint_func

    def extrapolate(self, trials: Mapping[str, Trial], max_target_flops: int) -> tuple[dict[str, Any], float] | None:
        """Extrapolate best configuration to target FLOPs."""
        if not trials or len([t for t in trials.values() if t.report is not None]) < 2:
            return None
            
        self.adapt_search_space(trials=trials, max_evaluation_flops=max_target_flops)
        try:
            conf, pred = self.bayesian_optimizer.extrapolate(
                trials, budget_info=BudgetInfo(cost_to_spend=max_target_flops)
            )
            if isinstance(conf, Sequence):
                conf = conf[0]
            return conf, pred
        except Exception as e:
            logger.warning(f"Could not extrapolate: {e}")
            return None
    
    @classmethod
    def get_power_law_curvature(cls, trials: Mapping[str, Trial]) -> tuple[float, float]:
        """Fit simple power law scaling: N_opt = C * (FLOPs)^alpha"""
        flops_list = []
        n_opt_list = []
        param_to_loss_flop = {}
        
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            try:
                flops = cls.flops_estimator(**trial.config)
                n_opt = cls.params_estimator(**trial.config)
                objective = trial.report.objective_to_minimize
            except Exception:
                logger.error(f"Could not compute FLOPs or N_opt for trial {trial.id}, skipping.")
                continue
            param_to_loss_flop[n_opt] = (flops, objective) if n_opt not in param_to_loss_flop or objective < param_to_loss_flop[n_opt][1] else param_to_loss_flop[n_opt]
        
        if len(param_to_loss_flop) < 2:
            raise ValueError("Not enough data points to fit scaling law.")
        
        n_opt_list = list(param_to_loss_flop.keys())
        flops_list = [val[0] for val in param_to_loss_flop.values()]
        log_flops = np.log(flops_list)
        log_n_opt = np.log(n_opt_list)
        A = np.vstack([log_flops, np.ones(len(log_flops))]).T
        alpha, log_C = np.linalg.lstsq(A, log_n_opt, rcond=None)[0]
        C = np.exp(log_C)
        return C, alpha

    def get_trial_artifacts(self, trials: Mapping[str, Trial] | None = None) -> list[Artifact] | None:
        """Return scaling law artifacts for runtime persistence.

        Consolidates artifacts from the base class (shared scaling plots) and 
        Bayesian Optimizer specific plots.

        Args:
            trials: Mapping of trial IDs to Trial objects. Required to generate plots.

        Returns:
            List of Artifact objects combining base class and BO-specific plots.
        """
        # Get base class scaling artifacts (shared plots)
        artifacts = super().get_trial_artifacts(trials) or []
        
        # Add Bayesian Optimizer specific artifacts
        if self.bayesian_optimizer is not None:
            try:
                bo_artifacts = self.bayesian_optimizer.get_trial_artifacts(trials)
                if bo_artifacts:
                    artifacts.extend(bo_artifacts)
            except Exception as e:
                logger.warning(f"Failed to generate BO artifacts: {e}")
        
        return artifacts
