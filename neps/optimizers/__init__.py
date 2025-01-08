from collections.abc import Callable, Mapping
from functools import partial

from neps.optimizers.base_optimizer import BaseOptimizer
from neps.optimizers.bayesian_optimization.optimizer import BayesianOptimization
from neps.optimizers.grid_search.optimizer import GridSearch
from neps.optimizers.multi_fidelity import (
    ASHA,
    AsyncHB,
    HyperBand,
    SuccessiveHalving,
)

# from neps.optimizers.multi_fidelity.hyperband_old import MOBSTER
from neps.optimizers.multi_fidelity.ifbo import IFBO
from neps.optimizers.random_search.optimizer import RandomSearch

# TODO: Rename Searcher to Optimizer...
SearcherMapping: Mapping[str, Callable[..., BaseOptimizer]] = {
    # BO kind
    "bayesian_optimization": partial(BayesianOptimization, use_priors=False),
    "pibo": partial(BayesianOptimization, use_priors=True),
    # Successive Halving
    "successive_halving": SuccessiveHalving,
    "successive_halving_prior": partial(SuccessiveHalving, sampler="prior"),
    # Hyperband
    "hyperband": HyperBand,
    "hyperband_prior": partial(HyperBand, sampler="prior"),
    # ASHA
    "asha": ASHA,
    "asha_prior": partial(ASHA, sampler="prior"),
    # AsyncHB
    "async_hb": AsyncHB,
    "async_hb_prior": partial(AsyncHB, sampler="prior"),
    # Priorband
    "priorband": partial(HyperBand, sampler="priorband"),
    "priorband_sh": partial(SuccessiveHalving, sampler="priorband"),
    "priorband_asha": partial(ASHA, sampler="priorband"),
    "priorband_async": partial(AsyncHB, sampler="priorband"),
    "successive_halving_priorband": partial(SuccessiveHalving, sampler="priorband"),
    # Model based hyperband
    # "mobster": MOBSTER,
    # "priorband_bo": partial(PriorBand, model_based=True),
    # Other
    "random_search": RandomSearch,
    "grid_search": GridSearch,
    "ifbo": IFBO,
}
