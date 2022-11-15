from __future__ import annotations

from functools import partial
from typing import Callable

from .base_acquisition import BaseAcquisition
from .cost_cooling import CostCooler
from .ei import ComprehensiveExpectedImprovement
from .prior_weighted import DecayingPriorWeightedAcquisition

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
    "cost_cooler": CostCooler,
}
