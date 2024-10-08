from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING

from .base_optimizer import BaseOptimizer
from .bayesian_optimization.optimizer import BayesianOptimization
from .grid_search.optimizer import GridSearch
from .multi_fidelity.hyperband import (
    MOBSTER,
    AsynchronousHyperband,
    Hyperband,
    HyperbandCustomDefault,
)
from .multi_fidelity.ifbo import IFBO
from .multi_fidelity.successive_halving import (
    AsynchronousSuccessiveHalving,
    AsynchronousSuccessiveHalvingWithPriors,
    SuccessiveHalving,
    SuccessiveHalvingWithPriors,
)
from .multi_fidelity_prior.async_priorband import PriorBandAsha, PriorBandAshaHB
from .multi_fidelity_prior.priorband import PriorBand
from .random_search.optimizer import RandomSearch

# TODO: Rename Searcher to Optimizer...
SearcherMapping: Mapping[str, Callable[..., BaseOptimizer]] = {
    "bayesian_optimization": partial(BayesianOptimization, use_priors=False),
    "pibo": partial(BayesianOptimization, use_priors=True),
    "random_search": RandomSearch,
    "grid_search": GridSearch,
    "successive_halving": SuccessiveHalving,
    "successive_halving_prior": SuccessiveHalvingWithPriors,
    "asha": AsynchronousSuccessiveHalving,
    "hyperband": Hyperband,
    "asha_prior": AsynchronousSuccessiveHalvingWithPriors,
    "hyperband_custom_default": HyperbandCustomDefault,
    "priorband": PriorBand,
    "priorband_bo": partial(PriorBand, model_based=True),
    "priorband_asha": PriorBandAsha,
    "priorband_asha_hyperband": PriorBandAshaHB,
    "mobster": MOBSTER,
    "ifbo": IFBO,
}
