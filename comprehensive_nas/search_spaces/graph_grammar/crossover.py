import random
from typing import Callable, List, Tuple

from .cfg import Grammar


def simple_crossover(
    parent1: str,
    parent2: str,
    grammar: Grammar,
    patience: int = 50,
) -> Tuple[str, str]:
    return grammar.crossover(parent1=parent1, parent2=parent2, patience=patience)


def repetitive_search_space_crossover(
    base_parent: str,
    motif_parents: List[str],
    base_grammar: Grammar,  # pylint: disable=W0613
    motif_grammars: List[Grammar],
    inner_crossover_strategy: Callable,
    motif_prefix: str = "M",
):
    def _motifs_in_base_tree(base_parent, motif_grammars, motif_prefix):
        return [
            i
            for i in range(1, len(motif_grammars) + 1)
            if f"{motif_prefix}{i}" in base_parent
        ]

    def swap(list1, list2, index1, index2):
        tmp = list1[index1]
        list1[index1] = list2[index2]
        list2[index2] = tmp
        return list1, list2

    child1_string_trees = [base_parent[0]] + motif_parents[0]
    child2_string_trees = [base_parent[1]] + motif_parents[1]
    parent1_potential_motif_candidates = _motifs_in_base_tree(
        base_parent[0], motif_grammars, motif_prefix
    )
    parent2_potential_motif_candidates = _motifs_in_base_tree(
        base_parent[1], motif_grammars, motif_prefix
    )

    random_draw = random.randint(
        0,
        min(
            len(parent1_potential_motif_candidates),
            len(parent2_potential_motif_candidates),
        ),
    )
    if random_draw == 0:
        # NOTE: only works for 1 hierarchical level!
        parent1_motifs = _motifs_in_base_tree(
            child1_string_trees[0], motif_grammars, motif_prefix
        )
        parent2_motifs = _motifs_in_base_tree(
            child2_string_trees[0], motif_grammars, motif_prefix
        )
        parent1_drawn_motif = random.choice(parent1_motifs)
        parent2_drawn_motif = random.choice(parent2_motifs)

        child1_string_trees, child2_string_trees = swap(
            child1_string_trees,
            child2_string_trees,
            parent1_drawn_motif,
            parent2_drawn_motif,
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
