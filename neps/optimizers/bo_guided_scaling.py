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
    Literal,
)

import numpy as np

from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
from neps.optimizers.optimizer import Artifact
from neps.optimizers.bayesian_optimization import BayesianOptimization
from neps.exceptions import ConstraintViolationError

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
                 max_target_flops: int,
                 sampling_strategy: Literal[
                     "space_expansion", 
                     "const_cost", 
                     "left_budget_contraction",
                    ] = "space_expansion",
                 ) -> None:
        """Initialize the Bayesian Optimization guided scaling optimizer.

        Args:
            space: The search space to use, without the fidelity.
            bayesian_optimizer: The Bayesian Optimization optimizer instance.
            flops_estimator: Function to estimate FLOPs for a configuration.
            params_estimator: Function to estimate parameters for a configuration.
            seen_datapoints_estimator: Function to estimate data points for a configuration.
            max_evaluation_flops: Maximum FLOPs allowed for initial evaluation.
            max_target_flops: Target maximum FLOPs for extrapolation (used to calculate max_expansion).
        """
        self.space = space
        self.flops_estimator = flops_estimator
        self.params_estimator = params_estimator
        self.seen_datapoints_estimator = seen_datapoints_estimator
        self.bayesian_optimizer = bayesian_optimizer
        self.max_target_flops = max_target_flops
        self.max_evaluation_flops = max_evaluation_flops
        
        # Calculate max_expansion based on the ratio between \
        # max_target_flops and max_evaluation_flops
        if  max_evaluation_flops <= max_target_flops:
            self.max_expansion = 1
        else:
            import math
            self.max_expansion = max(1, math.ceil(math.log2(max_evaluation_flops/max_target_flops)))
        print(f"max_expansion {self.max_expansion}")
        
        self._scaling_law_params: ScalingLawParameters | None = None
        self._best_candidate: dict[str, Any] | None = None
        self._best_loss: float | None = None
        
        self._total_budget_for_initial_design = self._compute_initial_design_budget()
        print(f"total initial {self._total_budget_for_initial_design:e}")
        
        self.bayesian_optimizer.cost_estimator = flops_estimator

        self.sampling_strategy = sampling_strategy
        
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

    def _compute_initial_design_budget(self) -> float:
        """Compute budget for initial design phase in space_expansion strategy.
        
        Uses geometric series: max_evaluation_flops * (1 - (1/2 + 1/4 + ... + 1/2^max_expansion))
        = max_evaluation_flops / 2^max_expansion
        """
        # Geometric series: sum of 1/2^i for i in [1, max_expansion]
        return self.max_evaluation_flops / 2 ** self.max_expansion

    def __call__(self, trials, budget_info=None, n=None):
        spent = sum([self.flops_estimator(**trial.config) for trial in trials.values()])
        if self.sampling_strategy == "const_cost":
            to_spend = self.max_target_flops
        elif self.sampling_strategy == "left_budget_contraction":
            if len(trials) < self.bayesian_optimizer.n_initial_design:
                to_spend = self.max_evaluation_flops/(10* self.bayesian_optimizer.n_initial_design)
            else:
                to_spend = self.max_evaluation_flops - spent
        elif self.sampling_strategy == "space_expansion":
            expansion_level = self._calculate_expansion_level(spent)
            print(f"expansion_level {expansion_level}")
            to_spend = self._get_space_expansion_budget(spent, expansion_level)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        print(f"BO_Guided_Scaling: to spend {to_spend:e} FLOPs, spent {spent:e} FLOPs")
        if to_spend <= 0:
            logger.warning("No remaining FLOPs budget for evaluation. Returning None.")
            return None
        self.adapt_search_space(max_flops=to_spend, around_max_flops=(self.sampling_strategy == "const_cost"))
        
        try:
            sample = self.bayesian_optimizer(
                trials, budget_info=BudgetInfo(
                    cost_to_spend=to_spend + spent, used_cost_budget=spent,
                    ), 
                n=n,
            )
            return sample
        except ConstraintViolationError as e:
            # If space_expansion strategy and we hit an exhaustion error, expand search space
            if self.sampling_strategy == "space_expansion":
                expansion_level = self._calculate_expansion_level(spent)
                print(f"expansion_level {expansion_level}")
                expanded_level = expansion_level + 1
                logger.warning(
                    f"Hit exhaustion error in space_expansion at level {expansion_level}: {e}. "
                    f"Retrying at expanded level {expanded_level}."
                )
                # Retry with expanded search space
                to_spend = self._get_space_expansion_budget(spent, expanded_level)
                self.adapt_search_space(max_flops=to_spend)
                return self.bayesian_optimizer(
                    trials, budget_info=BudgetInfo(
                        cost_to_spend=to_spend + spent, used_cost_budget=spent,
                        ), 
                    n=n,
                )
            raise e

    def _calculate_expansion_level(self, spent: float) -> int:
        """Calculate which expansion level we're at based on budget spent.
        
        Level 0: Initial design phase (0 to _total_budget_for_initial_design)
        Level i (i >= 1): Expansion phases where level i allows max_evaluation_flops / 2^i
        
        Args:
            spent: Total FLOPs spent so far
            
        Returns:
            Current expansion level (0, 1, 2, ...)
        """
        if spent < self._total_budget_for_initial_design:
            return 0
        
        # We're in expansion phases
        spent_in_expansion = spent - self._total_budget_for_initial_design
        
        # Level i uses budget of max_evaluation_flops / 2^i
        # Find which level we're in based on cumulative spending
        level = 1
        while level <= self.max_expansion:
            budget_for_level = self.max_evaluation_flops / (2 ** level)
            if spent_in_expansion < budget_for_level:
                return level
            spent_in_expansion -= budget_for_level
            level += 1
        
        # Beyond max_expansion, return max_expansion
        return self.max_expansion

    def _get_space_expansion_budget(self, spent: float, expansion_level: int) -> float:
        """Compute budget allocation for space_expansion strategy (stateless).
        
        Strategy:
        - Level 0 (Initial design): spend up to _total_budget_for_initial_design
        - Level i (i >= 1): max_evaluation_flops / 2^i
        
        Args:
            spent: Total FLOPs spent so far
            expansion_level: Current expansion level (0 = initial design, 1+ = expansion rounds)
            
        Returns:
            Budget (max FLOPs) for this sampling round
        """
        if expansion_level == 0:
            # Initial design phase
            print(
                f"Initial {self._total_budget_for_initial_design:e} spent {spent:e} FLOPs")
            if spent < self._total_budget_for_initial_design:
                per_trial = self._total_budget_for_initial_design / self.bayesian_optimizer.n_initial_design
                return per_trial
        else:
            return self.max_evaluation_flops / (2 ** expansion_level)
    
    def adapt_search_space(
        self,
        max_flops: int,
        around_max_flops: bool = False,
    ) -> None:
        """Adapt the search space constraint based on remaining budget."""
        def constraint_func(conf: Mapping[str, Any]) -> float:
            flops = self.flops_estimator(**conf)
            if around_max_flops:
                return 1 if max_flops - flops <= 0.1 * max_flops and flops <= 1.1 * max_flops else -1.0
            return max_flops - flops

        if max_flops <= 0:
            logger.warning("Max FLOPs is non-positive. No configurations will be valid.")
            raise ValueError("Budget is exhausted")
        self.bayesian_optimizer.constraints_func = constraint_func

    def extrapolate(self, trials: Mapping[str, Trial], max_target_flops: int) -> tuple[dict[str, Any], float] | None:
        """Extrapolate best configuration to target FLOPs."""
        if not trials or len([t for t in trials.values() if t.report is not None]) < 2:
            return None
        
        self.adapt_search_space(max_flops=max_target_flops)
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
