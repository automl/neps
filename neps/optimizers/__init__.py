from __future__ import annotations

from typing import Callable

from .base_optimizer import BaseOptimizer
from .bayesian_optimization.cost_cooling import CostCooling
from .bayesian_optimization.multi_fidelity import (
    BaseMultiFidelityOptimization,
    BayesianOptimizationMultiFidelity,
)
from .bayesian_optimization.optimizer import BayesianOptimization
from .grid_search.optimizer import GridSearch
from .random_search.optimizer import RandomSearch
from .regularized_evolution.optimizer import RegularizedEvolution

SearcherMapping: dict[str, Callable] = {
    "bayesian_optimization": BayesianOptimization,
    "mf_bayesian_optimization": BayesianOptimizationMultiFidelity,
    "random_search": RandomSearch,
    "cost_cooling_bayesian_optimization": CostCooling,
    "regularized_evolution": RegularizedEvolution,
    "grid_search": GridSearch,
}
