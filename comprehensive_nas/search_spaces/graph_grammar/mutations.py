import random
from typing import Callable, List, Tuple

from .cfg import Grammar


def simple_mutate(parent_string_tree: str, grammar: Grammar) -> Tuple[str, bool]:
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
    mutation_prob: float = None,
    patience: int = 50,
) -> Tuple[str, bool]:
    split_tree = parent_string_tree.split(" ")
    swappable_indices = [
        i
        for i in range(0, len(split_tree))
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
                for i in range(0, len(split_tree))
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
    motif_parents: List[str],
    base_grammar: Grammar,
    motif_grammars: List[Grammar],
    inner_mutation_strategy: Callable,
    mutation_rate: float = 1.0,
    mutation_prob: float = None,
) -> Tuple[List[str], List[bool]]:
    def _motifs_in_base_tree(base_parent, motif_grammars):
        return [
            i
            for i, grammar in enumerate(motif_grammars)
            if str(grammar.start()) in base_parent
        ]

    assert len(motif_parents) == len(motif_grammars)

    indices = _motifs_in_base_tree(base_parent, motif_grammars)
    mutation_prob = (
        mutation_rate / (len(indices) + 1) if mutation_prob is None else mutation_prob
    )

    child_string_trees = []
    if random.random() < mutation_prob:
        child_string_trees.append(inner_mutation_strategy(base_parent, base_grammar))
        indices = _motifs_in_base_tree(base_parent, motif_grammars)
        mutation_prob = (
            mutation_rate / (len(indices) + 1) if mutation_prob is None else mutation_prob
        )
    else:
        child_string_trees.append((base_parent, True))

    child_string_trees.extend(
        [
            inner_mutation_strategy(parent_string_tree, grammar)
            if i in indices and random.random() < mutation_prob
            else (parent_string_tree, True)
            for i, (parent_string_tree, grammar) in enumerate(
                zip(motif_parents, motif_grammars)
            )
        ]
    )

    return [c[0] for c in child_string_trees], [c[1] for c in child_string_trees]
