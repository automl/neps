from __future__ import annotations  # noqa: D100

import random
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .cfg import Grammar


def simple_mutate(parent_string_tree: str, grammar: Grammar) -> tuple[str, bool]:  # noqa: D103
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


def bananas_mutate(  # noqa: D103
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
        if random.random() < _mutation_prob:  # noqa: S311
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
