

from functools import partial
from typing import Callable, Mapping

from .base_optimizer import BaseOptimizer
from .bayesian_optimization.cost_cooling import CostCooling
from .bayesian_optimization.optimizer import BayesianOptimization
from .grid_search.optimizer import GridSearch
from .multi_fidelity.ifbo import IFBO
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
from .random_search.optimizer import RandomSearch
from .regularized_evolution.optimizer import RegularizedEvolution

# TODO: Rename Searcher to Optimizer...
SearcherMapping: Mapping[str, Callable[..., BaseOptimizer]] = {
    "bayesian_optimization": BayesianOptimization,
    "pibo": partial(BayesianOptimization, disable_priors=False),
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
    "hyperband_custom_default": HyperbandCustomDefault,
    "priorband": PriorBand,
    "priorband_bo": partial(PriorBand, model_based=True),
    "priorband_asha": PriorBandAsha,
    "priorband_asha_hyperband": PriorBandAshaHB,
    "mobster": MOBSTER,
    "ifbo": IFBO,
}
