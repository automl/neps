from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .cfg import Grammar


def simple_crossover(
    parent1: str,
    parent2: str,
    grammar: Grammar,
    patience: int = 50,
    return_crossover_subtrees: bool = False,
) -> tuple[str, str]:
    if return_crossover_subtrees:
        return grammar.crossover(
            parent1=parent1,
            parent2=parent2,
            patience=patience,
            return_crossover_subtrees=return_crossover_subtrees,
        )
    return grammar.crossover(
        parent1=parent1,
        parent2=parent2,
        patience=patience,
    )
