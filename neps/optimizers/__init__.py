from __future__ import annotations

from functools import partial
from typing import Callable

from .base_optimizer import BaseOptimizer
from .bayesian_optimization.optimizer import \
    BayesianOptimization, BayesianOptimizationMultiFidelity
from .random_search.optimizer import RandomSearch

SearcherMapping: dict[str, Callable] = {
    "bayesian_optimization": BayesianOptimization,
    "mf_bayesian_optimization": BayesianOptimizationMultiFidelity,
    "random_search": RandomSearch,
}
