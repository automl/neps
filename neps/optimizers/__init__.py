from __future__ import annotations

from typing import Callable

from .base_optimizer import BaseOptimizer
from .bayesian_optimization.acquisition_functions.ei_cost_aware import CArBO
from .bayesian_optimization.optimizer import BayesianOptimization
from .random_search.optimizer import RandomSearch

SearcherMapping: dict[str, Callable] = {
    "bayesian_optimization": BayesianOptimization,
    "random_search": RandomSearch,
    "CArBO": CArBO,  # TODO: partial(BayesianOptimization, acquisition_function="CArBO")
}
