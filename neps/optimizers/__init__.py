from collections.abc import Callable, Mapping
from functools import partial

from neps.optimizers.base_optimizer import BaseOptimizer
from neps.optimizers.bayesian_optimization.optimizer import BayesianOptimization
from neps.optimizers.grid_search.optimizer import GridSearch
from neps.optimizers.multi_fidelity import IFBO, MOBSTER, Hyperband
from neps.optimizers.multi_fidelity.asha import ASHA
from neps.optimizers.multi_fidelity.successive_halving import SuccessiveHalving
from neps.optimizers.multi_fidelity_prior import (
    PriorBand,
    PriorBandAsha,
    PriorBandAshaHB,
)
from neps.optimizers.random_search.optimizer import RandomSearch

# TODO: Rename Searcher to Optimizer...
SearcherMapping: Mapping[str, Callable[..., BaseOptimizer]] = {
    # BO kind
    "bayesian_optimization": partial(BayesianOptimization, use_priors=False),
    "pibo": partial(BayesianOptimization, use_priors=True),
    # Successive Halving kind
    "successive_halving": SuccessiveHalving,
    "successive_halving_prior": partial(SuccessiveHalving, sampler="prior"),
    "asha": ASHA,
    "asha_prior": partial(ASHA, sampler="prior"),
    "priorband_asha": PriorBandAsha,
    # Hyperband kind
    "hyperband": Hyperband,
    "priorband": PriorBand,
    "priorband_asha_hyperband": PriorBandAshaHB,
    # Model based hyperband
    "mobster": MOBSTER,
    "priorband_bo": partial(PriorBand, model_based=True),
    # Other
    "random_search": RandomSearch,
    "grid_search": GridSearch,
    "ifbo": IFBO,
}
