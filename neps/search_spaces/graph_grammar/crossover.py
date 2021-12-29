import random
from copy import deepcopy
from typing import Callable, List, Tuple

from .cfg import Grammar


def simple_crossover(
    parent1: str,
    parent2: str,
    grammar: Grammar,
    patience: int = 50,
    return_crossover_subtrees: bool = False,
) -> Tuple[str, str]:
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
    base_parent: Tuple[str, str],
    motif_parents: Tuple[List[str], List[str]],
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
    if random_draw == 0:  # crossover high level grammar
        parent1_motifs = _motifs_in_base_tree(
            child1_string_trees[0], motif_grammars, motif_prefix
        )
        parent2_motifs = _motifs_in_base_tree(
            child2_string_trees[0], motif_grammars, motif_prefix
        )
        (_, _, subtrees_child1, subtrees_child2,) = inner_crossover_strategy(
            child1_string_trees[0],
            child2_string_trees[0],
            base_grammar,
            return_crossover_subtrees=True,
        )
        subtrees_child1 = list(subtrees_child1)
        subtrees_child2 = list(subtrees_child2)
        new_child1_motifs = _motifs_in_base_tree(
            subtrees_child2[1], motif_grammars, motif_prefix
        )
        new_child2_motifs = _motifs_in_base_tree(
            subtrees_child1[1], motif_grammars, motif_prefix
        )

        old_child1_string_trees = deepcopy(child1_string_trees)
        free_motifs = list(set(range(1, len(motif_grammars) + 1)) - set(parent1_motifs))
        if len(free_motifs) > 0:
            if len(new_child1_motifs) > len(free_motifs):  # too many new child motifs
                new_child1_motifs = random.sample(
                    new_child1_motifs,
                    k=len(free_motifs),
                )
            elif len(new_child1_motifs) < len(
                free_motifs
            ):  # more free spots than necessary
                free_motifs = random.sample(
                    free_motifs,
                    k=len(new_child1_motifs),
                )
            for fm, nm in zip(free_motifs, new_child1_motifs):
                child1_string_trees[fm] = child2_string_trees[nm].replace(
                    f"{motif_prefix}{nm}", f"{motif_prefix}{fm}"
                )
                subtrees_child2[1] = subtrees_child2[1].replace(
                    f"{motif_prefix}{nm}", f"{motif_prefix}{fm}"
                )
        child1_string_trees[0] = (
            subtrees_child1[0] + subtrees_child2[1] + subtrees_child1[2]
        )

        free_motifs = list(set(range(1, len(motif_grammars) + 1)) - set(parent2_motifs))
        if len(free_motifs) > 0:
            if len(new_child2_motifs) > len(free_motifs):
                new_child2_motifs = random.sample(
                    new_child2_motifs,
                    k=len(free_motifs),
                )
            elif len(new_child2_motifs) < len(free_motifs):
                free_motifs = random.sample(
                    free_motifs,
                    k=len(new_child2_motifs),
                )
            for fm, nm in zip(free_motifs, new_child2_motifs):
                child2_string_trees[fm] = old_child1_string_trees[nm].replace(
                    f"{motif_prefix}{nm}", f"{motif_prefix}{fm}"
                )
                subtrees_child1[1] = subtrees_child1[1].replace(
                    f"{motif_prefix}{nm}", f"{motif_prefix}{fm}"
                )
        child2_string_trees[0] = (
            subtrees_child2[0] + subtrees_child1[1] + subtrees_child2[2]
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
