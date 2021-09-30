from typing import Tuple

from .cfg import Grammar


def crossover(
    parent1: str,
    parent2: str,
    grammar: Grammar,
    patience: int = 50,
) -> Tuple[str, str]:
    # randomly swap subtrees in two trees
    # if no suitiable subtree exists then return False
    subtree_node, subtree_index = grammar.rand_subtree(parent1)
    # chop out subtree
    pre, sub, post = grammar.remove_subtree(parent1, subtree_index)
    _patience = patience
    while _patience > 0:
        # sample subtree from donor
        if grammar.is_depth_constrained():
            head_node_constraint = grammar.compute_depth_information_for_pre(pre)[
                subtree_node
            ]
            donor_subtree_index = grammar.rand_subtree_fixed_head(
                parent2, subtree_node, head_node_constraint
            )
        else:
            donor_subtree_index = grammar.rand_subtree_fixed_head(parent2, subtree_node)
        # if no subtrees with right head node return False
        if not donor_subtree_index:
            _patience -= 1
        else:
            donor_pre, donor_sub, donor_post = grammar.remove_subtree(
                parent2, donor_subtree_index
            )
            # return the two new tree
            child1 = pre + donor_sub + post
            child2 = donor_pre + sub + donor_post
            return child1, child2

    return False, False
