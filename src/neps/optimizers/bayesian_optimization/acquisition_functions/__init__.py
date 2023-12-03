from __future__ import annotations

from functools import partial
from typing import Callable

from .ei import ComprehensiveExpectedImprovement
from .mf_ei import MFEI, MFEI_AtMax, MFEI_Dyna
from .ucb import UpperConfidenceBound
from .mf_ucb import MF_UCB, MF_UCB_AtMax, MF_UCB_Dyna


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
    "MFEI-max": partial(
        MFEI_AtMax,
        in_fill="best",
        augmented_ei=False,
    ),
    "MFEI-dyna": partial(
        MFEI_Dyna,
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
    "MF-UCB-max": partial(
        MF_UCB_AtMax,
        maximize=False,
    ),
    "MF-UCB-dyna": partial(
        MF_UCB_Dyna,
        maximize=False,
    ),
}
