from typing import Tuple

from .cfg import Grammar


def simple_crossover(
    parent1: str,
    parent2: str,
    grammar: Grammar,
    patience: int = 50,
) -> Tuple[str, str]:
    return grammar.crossover(parent1=parent1, parent2=parent2, patience=patience)
