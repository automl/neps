from __future__ import annotations

from typing import Callable

from .base_optimizer import BaseOptimizer
from .bayesian_optimization.cost_cooling import CostCooling
from .bayesian_optimization.mf_tpe import MultiFidelityPriorWeightedTreeParzenEstimator
from .bayesian_optimization.multi_fidelity import BayesianOptimizationMultiFidelity
from .bayesian_optimization.optimizer import BayesianOptimization
from .grid_search.optimizer import GridSearch
from .multi_fidelity.successive_halving import (
    AsynchronousSuccessiveHalving,
    AsynchronousSuccessiveHalvingWithPriors,
    SuccessiveHalving,
    SuccessiveHalvingWithPriors,
)

# custom algorithms
from .multi_fidelity_prior.v1 import OurOptimizerV1, OurOptimizerV1_2, OurOptimizerV1_3
from .multi_fidelity_prior.v2 import OurOptimizerV2, OurOptimizerV2_2, OurOptimizerV2_3
from .random_search.optimizer import RandomSearch
from .regularized_evolution.optimizer import RegularizedEvolution

SearcherMapping: dict[str, Callable] = {
    "bayesian_optimization": BayesianOptimization,
    "mf_bayesian_optimization": BayesianOptimizationMultiFidelity,
    "cost_cooling_bayesian_optimization": CostCooling,
    "random_search": RandomSearch,
    "cost_cooling": CostCooling,
    "regularized_evolution": RegularizedEvolution,
    "grid_search": GridSearch,
    "successive_halving": SuccessiveHalving,
    "successive_halving_prior": SuccessiveHalvingWithPriors,
    "asha": AsynchronousSuccessiveHalving,
    "asha_prior": AsynchronousSuccessiveHalvingWithPriors,
    "multifidelity_tpe": MultiFidelityPriorWeightedTreeParzenEstimator,
    # custom algorithms
    "ours_v1": OurOptimizerV1,
    "ours_v1_2": OurOptimizerV1_2,
    "ours_v1_3": OurOptimizerV1_3,
    "ours_v2": OurOptimizerV2,
    "ours_v2_2": OurOptimizerV2_2,
    "ours_v2_3": OurOptimizerV2_3,
}
