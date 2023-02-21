from __future__ import annotations

from functools import partial
from typing import Callable

from .base_optimizer import BaseOptimizer
from .bayesian_optimization.cost_cooling import CostCooling
from .bayesian_optimization.mf_tpe import MultiFidelityPriorWeightedTreeParzenEstimator
from .bayesian_optimization.optimizer import BayesianOptimization
from .grid_search.optimizer import GridSearch
from .multi_fidelity.hyperband import (
    MOBSTER,
    AsynchronousHyperband,
    Hyperband,
    HyperbandCustomDefault,
)
from .multi_fidelity.successive_halving import (
    AsynchronousSuccessiveHalving,
    AsynchronousSuccessiveHalvingWithPriors,
    SuccessiveHalving,
    SuccessiveHalvingWithPriors,
)
from .multi_fidelity_prior.async_priorband import PriorBandAsha, PriorBandAshaHB
from .multi_fidelity_prior.priorband import PriorBand
from .multi_fidelity_prior.raceband import RaceBand
from .random_search.optimizer import RandomSearch
from .regularized_evolution.optimizer import RegularizedEvolution

SearcherMapping: dict[str, Callable] = {
    "bayesian_optimization": BayesianOptimization,
    # "mf_bayesian_optimization": BayesianOptimizationMultiFidelity,
    "cost_cooling_bayesian_optimization": CostCooling,
    "random_search": RandomSearch,
    "cost_cooling": CostCooling,
    "regularized_evolution": RegularizedEvolution,
    "assisted_regularized_evolution": partial(RegularizedEvolution, assisted=True),
    "grid_search": GridSearch,
    "successive_halving": SuccessiveHalving,
    "successive_halving_prior": SuccessiveHalvingWithPriors,
    "asha": AsynchronousSuccessiveHalving,
    "hyperband": Hyperband,
    "asha_prior": AsynchronousSuccessiveHalvingWithPriors,
    "multifidelity_tpe": MultiFidelityPriorWeightedTreeParzenEstimator,
    "raceband": RaceBand,
    "hyperband_custom_default": HyperbandCustomDefault,
    "priorband": PriorBand,
    "mobster": MOBSTER,
}
