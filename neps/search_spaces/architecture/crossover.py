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


def repetitive_search_space_crossover(
    base_parent: tuple[str, str],
    motif_parents: tuple[list[str], list[str]],
    base_grammar: Grammar,
    motif_grammars: list[Grammar],
    terminal_to_sublanguage_map: dict,
    number_of_repetitive_motifs_per_grammar: list,
    inner_crossover_strategy: Callable,
    fixed_macro_parent: bool = False,
    multiple_repetitive: bool = False,
):
    def _motifs_in_base_tree(base_parent, terminal_to_sublanguage_map):
        return [
            i + 1
            for i, k in enumerate(terminal_to_sublanguage_map.keys())
            if k in base_parent
        ]

    child1_string_trees = [base_parent[0]] + motif_parents[0]
    child2_string_trees = [base_parent[1]] + motif_parents[1]
    parent1_potential_motif_candidates = _motifs_in_base_tree(
        base_parent[0], terminal_to_sublanguage_map
    )
    parent2_potential_motif_candidates = _motifs_in_base_tree(
        base_parent[1], terminal_to_sublanguage_map
    )

    random_draw = random.randint(  # noqa: S311
        1 if fixed_macro_parent else 0,
        min(
            len(parent1_potential_motif_candidates),
            len(parent2_potential_motif_candidates),
        ),
    )
    if random_draw == 0:  # crossover high level grammar, but keep repetitive motifs fixed
        (
            _,
            _,
            subtrees_child1,
            subtrees_child2,
        ) = inner_crossover_strategy(
            child1_string_trees[0],
            child2_string_trees[0],
            base_grammar,
            return_crossover_subtrees=True,
        )
        subtrees_child1 = list(subtrees_child1)
        subtrees_child2 = list(subtrees_child2)
        child1_string_trees[0] = (
            subtrees_child1[0] + subtrees_child2[1] + subtrees_child1[2]
        )

        child2_string_trees[0] = (
            subtrees_child2[0] + subtrees_child1[1] + subtrees_child2[2]
        )
    elif multiple_repetitive:
        # TODO more general procedure
        coin_toss = random.randint(1, len(child1_string_trees) - 1)
        motif_grammar_idx = next(
            i
            for i, x in enumerate(np.cumsum(number_of_repetitive_motifs_per_grammar))
            if x >= coin_toss
        )
        (
            child1_string_trees[coin_toss],
            child2_string_trees[coin_toss],
        ) = inner_crossover_strategy(
            child1_string_trees[coin_toss],
            child2_string_trees[coin_toss],
            motif_grammars[motif_grammar_idx],
        )
    else:
        parent1_random_draw = random.randint(
            0, len(parent1_potential_motif_candidates) - 1
        )
        parent2_random_draw = random.randint(
            0, len(parent2_potential_motif_candidates) - 1
        )
        (
            child1_string_trees[parent1_random_draw + 1],
            child2_string_trees[parent2_random_draw + 1],
        ) = inner_crossover_strategy(
            child1_string_trees[parent1_random_draw + 1],
            child2_string_trees[parent2_random_draw + 1],
            motif_grammars[0],
        )

    if any(not st for st in child1_string_trees) or any(
        not st for st in child2_string_trees
    ):
        return False, False
    return child1_string_trees, child2_string_trees
