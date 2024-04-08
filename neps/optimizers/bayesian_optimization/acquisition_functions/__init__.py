from __future__ import annotations

from functools import partial
from typing import Callable

from .ei import ComprehensiveExpectedImprovement
from .mf_ei import MFEI, MFEI_AtMax, MFEI_Dyna, MFEI_Random
from .mf_pi import MFPI, MFPI_AtMax, MFPI_Dyna, MFPI_Random, MFPI_Random_HiT
from .mf_two_step import MF_TwoStep
from .mf_ucb import MF_UCB, MF_UCB_AtMax, MF_UCB_Dyna
from .ucb import UpperConfidenceBound

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
    "MFEI-random": partial(
        MFPI_Random,  # code has been modified, rerun and use "MFEI-random2"!
        in_fill="best",
        augmented_ei=False,
    ),
    "MFEI-random2": partial(
        MFEI_Random,
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
    "MF_TwoStep": partial(
        MF_TwoStep,
        maximize=False,
    ),
    "MFPI": partial(
        MFPI,
        in_fill="best",
        augmented_ei=False,
    ),
    "MFPI-max": partial(
        MFPI_AtMax,
        in_fill="best",
        augmented_ei=False,
    ),
    "MFPI-thresh-max": partial(
        MFPI_Random,
        in_fill="best",
        augmented_ei=False,
        horizon="max",
        threshold="random",
    ),
    "MFPI-random-horizon": partial(
        MFPI_Random,
        in_fill="best",
        augmented_ei=False,
        horizon="random",
        threshold="0.0",
    ),
    "MFPI-dyna": partial(
        MFPI_Dyna,
        in_fill="best",
        augmented_ei=False,
    ),
    "MFPI-random": partial(
        MFPI_Random,
        in_fill="best",
        augmented_ei=False,
    ),
    "MFPI-random-hit": partial(
        MFPI_Random_HiT,
        in_fill="best",
        augmented_ei=False,
    ),
}
