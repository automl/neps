from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable, Mapping

from .base_optimizer import BaseOptimizer
from .bayesian_optimization.optimizer import BayesianOptimization
from .grid_search.optimizer import GridSearch
from .multi_fidelity.dyhpo import MFEIBO
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
from .multi_fidelity_prior.priorband import PriorBand
from .random_search.optimizer import RandomSearch
from .regularized_evolution.optimizer import RegularizedEvolution

if TYPE_CHECKING:
    from .base_optimizer import BaseOptimizer

# TODO: Rename Searcher to Optimizer...
SearcherMapping: Mapping[str, Callable[..., BaseOptimizer]] = {
    "bayesian_optimization": partial(BayesianOptimization, use_priors=False),
    "pibo": partial(BayesianOptimization, use_priors=True),
    "random_search": RandomSearch,
    "regularized_evolution": RegularizedEvolution,
    "assisted_regularized_evolution": partial(RegularizedEvolution, assisted=True),
    "grid_search": GridSearch,
    "successive_halving": SuccessiveHalving,
    "successive_halving_prior": SuccessiveHalvingWithPriors,
    "asha": AsynchronousSuccessiveHalving,
    "hyperband": Hyperband,
    "asha_prior": AsynchronousSuccessiveHalvingWithPriors,
    "hyperband_custom_default": HyperbandCustomDefault,
    "priorband": PriorBand,
    "mobster": MOBSTER,
    "mf_ei_bo": MFEIBO,
}
