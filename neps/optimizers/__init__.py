from collections.abc import Callable, Mapping
from functools import partial

from neps.optimizers.base_optimizer import BaseOptimizer
from neps.optimizers.bayesian_optimization.optimizer import BayesianOptimization
from neps.optimizers.grid_search.optimizer import GridSearch
from neps.optimizers.multi_fidelity import (
    IFBO,
    MOBSTER,
    AsynchronousSuccessiveHalving,
    AsynchronousSuccessiveHalvingWithPriors,
    Hyperband,
    HyperbandCustomDefault,
    SuccessiveHalving,
    SuccessiveHalvingWithPriors,
)
from neps.optimizers.multi_fidelity_prior import (
    PriorBand,
    PriorBandAsha,
    PriorBandAshaHB,
)
from neps.optimizers.random_search.optimizer import RandomSearch

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
