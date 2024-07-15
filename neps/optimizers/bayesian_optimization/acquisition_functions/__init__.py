

from functools import partial
from typing import Callable

from neps.optimizers.bayesian_optimization.acquisition_functions.ei import (
    ComprehensiveExpectedImprovement,
)
from neps.optimizers.bayesian_optimization.acquisition_functions.mf_ei import MFEI
from neps.optimizers.bayesian_optimization.acquisition_functions.mf_pi import MFPI_Random
from neps.optimizers.bayesian_optimization.acquisition_functions.ucb import (
    UpperConfidenceBound,
    MF_UCB,
)
from neps.optimizers.bayesian_optimization.acquisition_functions.prior_weighted import (
    DecayingPriorWeightedAcquisition,
)


AcquisitionMapping: dict[str, Callable] = {
    "EI": partial(
        ComprehensiveExpectedImprovement,
        in_fill="best",
        augmented_ei=False,
    ),
    "LogEI": partial(
        ComprehensiveExpectedImprovement,
        in_fill="best",
        augmented_ei=False,
        log_ei=True,
    ),
    #     # Uses the augmented EI heuristic and changed the in-fill criterion to the best test location with
    #     # the highest *posterior mean*, which are preferred when the optimisation is noisy.
    "AEI": partial(
        ComprehensiveExpectedImprovement,
        in_fill="posterior",
        augmented_ei=True,
    ),
    "MFEI": partial(
        MFEI,
        in_fill="best",
        augmented_ei=False,
    ),
    "MFPI-random": partial(
        MFPI_Random,
        in_fill="best",
        augmented_ei=False,
    ),
    "UCB": partial(
        UpperConfidenceBound,
        maximize=False,
    ),
    "MF-UCB": partial(
        MF_UCB,
        maximize=False,
    ),
}

__all__ = [
    "AcquisitionMapping",
    "ComprehensiveExpectedImprovement",
    "MFEI",
    "UpperConfidenceBound",
    "MF_UCB",
    "DecayingPriorWeightedAcquisition",
]
