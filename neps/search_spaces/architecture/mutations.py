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
    terminal_to_sublanguage_map: dict,
    number_of_repetitive_motifs_per_grammar: list,
    inner_mutation_strategy: Callable,
    mutation_rate: float = 1.0,
    mutation_prob: float = None,
    fixed_macro_parent: bool = False,
) -> Tuple[List[str], List[bool]]:
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
    # child_string_trees.extend(
    #     [
    #         inner_mutation_strategy(parent_string_tree, grammar)
    #         if i in indices and random.random() < mutation_prob
    #         else (parent_string_tree, True)
    #         for i, (parent_string_tree, grammar) in enumerate(
    #             zip(motif_parents, motif_grammars)
    #         )
    #     ]
    # )

    return [c[0] for c in child_string_trees], [c[1] for c in child_string_trees]
