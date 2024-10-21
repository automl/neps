from collections.abc import Callable, Mapping
from functools import partial

from neps.optimizers.base_optimizer import BaseOptimizer
from neps.optimizers.bayesian_optimization.optimizer import BayesianOptimization
from neps.optimizers.grid_search.optimizer import GridSearch
from neps.optimizers.multi_fidelity import (
    IFBO,
    MOBSTER,
    Hyperband,
    SuccessiveHalving,
)
from neps.optimizers.multi_fidelity.promotion_policy import (
    AsyncPromotionPolicy,
    SyncPromotionPolicy,
)
from neps.optimizers.multi_fidelity.sampling_policy import (
    FixedPriorPolicy,
    RandomUniformPolicy,
)
from neps.optimizers.multi_fidelity.successive_halving import SuccessiveHalvingBase
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
    "successive_halving_prior": partial(
        SuccessiveHalving,
        sampling_policy=FixedPriorPolicy,
        promotion_policy=SyncPromotionPolicy,
        use_priors=True,
        prior_confidence="medium",
    ),
    "hyperband": Hyperband,
    "asha": partial(
        SuccessiveHalvingBase,
        sampling_policy=RandomUniformPolicy,
        promotion_policy=AsyncPromotionPolicy,
        use_priors=False,
        prior_confidence=None,
    ),
    "asha_prior": partial(
        SuccessiveHalvingBase,
        sampling_policy=FixedPriorPolicy,
        promotion_policy=AsyncPromotionPolicy,
        use_priors=True,
        prior_confidence="medium",
    ),
    "priorband": PriorBand,
    "priorband_bo": partial(PriorBand, model_based=True),
    "priorband_asha": PriorBandAsha,
    "priorband_asha_hyperband": PriorBandAshaHB,
    "mobster": MOBSTER,
    "ifbo": IFBO,
}
