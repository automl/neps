from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .cfg import Grammar


def simple_mutate(parent_string_tree: str, grammar: Grammar) -> tuple[str, bool]:
    # works if there is only one grammar
    # randomly choose a subtree from the parent and replace
    # with a new randomly generated subtree

    # choose subtree to delete
    subtree_node, subtree_idx = grammar.rand_subtree(parent_string_tree)
    child_string_tree = grammar.mutate(
        parent=parent_string_tree,
        subtree_index=subtree_idx,
        subtree_node=subtree_node,
    )
    return child_string_tree, parent_string_tree == child_string_tree


def bananas_mutate(
    parent_string_tree: str,
    grammar: Grammar,
    mutation_rate: float = 1.0,
    mutation_prob: float | None = None,
    patience: int = 50,
) -> tuple[str, bool]:
    split_tree = parent_string_tree.split(" ")
    swappable_indices = [
        i
        for i in range(len(split_tree))
        if split_tree[i][1:] in grammar.swappable_nonterminals
    ]
    _mutation_prob = (
        mutation_rate / len(swappable_indices) if mutation_prob is None else mutation_prob
    )
    child_string_tree = parent_string_tree

    idx = 0
    while idx < len(swappable_indices):
        swap_idx = swappable_indices[idx]
        if random.random() < _mutation_prob:
            subtree_node = split_tree[swap_idx][1:]
            subtree_idx = swap_idx
            child_string_tree = grammar.mutate(
                parent=child_string_tree,
                subtree_index=subtree_idx,
                subtree_node=subtree_node,
                patience=patience,
            )

            # update swappable indices
            split_tree = child_string_tree.split(" ")
            swappable_indices = [
                i
                for i in range(len(split_tree))
                if split_tree[i][1:] in grammar.swappable_nonterminals
            ]
            _mutation_prob = (
                mutation_rate / len(swappable_indices)
                if mutation_prob is None
                else mutation_prob
            )
        idx += 1

    return child_string_tree, child_string_tree == parent_string_tree


def repetitive_search_space_mutation(
    base_parent: str,
    motif_parents: list[str],
    base_grammar: Grammar,
    motif_grammars: list[Grammar],
    terminal_to_sublanguage_map: dict,
    number_of_repetitive_motifs_per_grammar: list,
    inner_mutation_strategy: Callable,
    mutation_rate: float = 1.0,
    mutation_prob: float | None = None,
    fixed_macro_parent: bool = False,
) -> tuple[list[str], list[bool]]:
    def _motifs_in_base_tree(base_parent, terminal_to_sublanguage_map):
        return [
            i
            for i, k in enumerate(terminal_to_sublanguage_map.keys())
            if k in base_parent
        ]

    indices = _motifs_in_base_tree(base_parent, terminal_to_sublanguage_map)
    if fixed_macro_parent:
        mutation_prob = (
            mutation_rate / len(indices) if mutation_prob is None else mutation_prob
        )
    else:
        mutation_prob = (
            mutation_rate / (len(indices) + 1) if mutation_prob is None else mutation_prob
        )

    child_string_trees = []
    if not fixed_macro_parent and random.random() < mutation_prob:
        child_string_trees.append(inner_mutation_strategy(base_parent, base_grammar))
        indices = _motifs_in_base_tree(base_parent, terminal_to_sublanguage_map)
        mutation_prob = (
            mutation_rate / (len(indices) + 1) if mutation_prob is None else mutation_prob
        )
    else:
        child_string_trees.append((base_parent, True))

    parent_string_idx = 0
    _number_of_repetitive_motifs_per_grammar = (
        number_of_repetitive_motifs_per_grammar[1:]
        if not fixed_macro_parent
        else number_of_repetitive_motifs_per_grammar
    )
    for grammar, number_of_motifs in zip(
        motif_grammars, _number_of_repetitive_motifs_per_grammar
    ):
        for _ in range(number_of_motifs):
            if parent_string_idx in indices and random.random() < mutation_prob:
                child_string_trees.append(
                    inner_mutation_strategy(motif_parents[parent_string_idx], grammar)
                )
            else:
                child_string_trees.append((motif_parents[parent_string_idx], True))
            parent_string_idx += 1

    return [c[0] for c in child_string_trees], [c[1] for c in child_string_trees]
