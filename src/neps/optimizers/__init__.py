from __future__ import annotations

from typing import Callable

from .base_optimizer import BaseOptimizer
from .bayesian_optimization.cost_cooling import CostCooling
from .bayesian_optimization.mf_tpe import MultiFidelityPriorWeightedTreeParzenEstimator
from .bayesian_optimization.optimizer import BayesianOptimization
from .grid_search.optimizer import GridSearch
from .multi_fidelity.hyperband import Hyperband, HyperbandCustomDefault
from .multi_fidelity.successive_halving import (
    AsynchronousSuccessiveHalving,
    AsynchronousSuccessiveHalvingWithPriors,
    SuccessiveHalving,
    SuccessiveHalvingWithPriors,
)
from .multi_fidelity_prior.async_priorband import PriorBandAsha, PriorBandAshaHB
from .multi_fidelity_prior.priorband import PriorBand
from .multi_fidelity_prior.raceband import RaceBand

## custom algorithms
from .multi_fidelity_prior.v1 import OurOptimizerV1, OurOptimizerV1_2, OurOptimizerV1_3
from .multi_fidelity_prior.v2 import OurOptimizerV2, OurOptimizerV2_2, OurOptimizerV2_3
from .multi_fidelity_prior.v3 import OurOptimizerV3, OurOptimizerV3_2
from .multi_fidelity_prior.v4 import (
    OurOptimizerV4_ASHA,
    OurOptimizerV4_ASHA_HB,
    OurOptimizerV4_HB,
    OurOptimizerV4_SH,
    OurOptimizerV4_V3_2,
)
from .multi_fidelity_prior.v5 import (
    OurOptimizerV5,
    OurOptimizerV5_2,
    OurOptimizerV5_2_V4,
    OurOptimizerV5_3,
    OurOptimizerV5_V4,
)
from .multi_fidelity_prior.v6 import (
    OurOptimizerV6,
    OurOptimizerV6_V5,
    OurOptimizerV6_V5_2,
    OurOptimizerV6_V5_3,
)
from .random_search.optimizer import RandomSearch
from .regularized_evolution.optimizer import RegularizedEvolution

SearcherMapping: dict[str, Callable] = {
    "bayesian_optimization": BayesianOptimization,
    # "mf_bayesian_optimization": BayesianOptimizationMultiFidelity,
    "cost_cooling_bayesian_optimization": CostCooling,
    "random_search": RandomSearch,
    "cost_cooling": CostCooling,
    "regularized_evolution": RegularizedEvolution,
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
}
