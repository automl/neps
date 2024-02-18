from collections import deque
from typing import Deque

import networkx as nx
import numpy as np
from nltk.grammar import Nonterminal

from ..cfg import Grammar, choice


class ResolutionGrammar(Grammar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_downsamples = None
        self.terminal_to_graph_map = None
        self.downsampling_lhs = None
        self.downsample_terminal = None
        self.depth_constraints = None

    def set_resolution_constraints(
        self,
        n_downsamples: int,
        terminal_to_graph: dict,
        downsampling_lhs: str,
        downsample_terminal: str = "downsample",
        depth_constraints: dict = None,
    ):
        self.n_downsamples = n_downsamples

        terminal_to_graph_map: dict = {}
        for k, v in terminal_to_graph.items():
            terminal_to_graph_map[k] = {}
            terminal_to_graph_map[k]["edge_list"] = v

            G = nx.DiGraph()
            G.add_edges_from(v)
            src = [n for n, d in G.in_degree() if d == 0][0]
            tgt = [n for n, d in G.out_degree() if d == 0][0]
            terminal_to_graph_map[k]["paths"] = {
                k: [] for k in range(1, nx.dag_longest_path_length(G) + 1)
            }
            for path in nx.all_simple_edge_paths(G, source=src, target=tgt):
                terminal_to_graph_map[k]["paths"][len(path)].append(path[::-1])

        self.terminal_to_graph_map = terminal_to_graph_map

        self.downsampling_lhs = downsampling_lhs
        self.swappable_nonterminals.remove(self.downsampling_lhs)

        self.downsample_terminal = downsample_terminal

        if depth_constraints is not None:
            self.depth_constraints = depth_constraints
            if not all(k in self.nonterminals for k in self.depth_constraints.keys()):
                raise Exception(
                    f"Nonterminal {set(self.depth_constraints.keys())-set(self.nonterminals)} does not exist in grammar"
                )
        else:
            self.depth_constraints = {}

    @staticmethod
    def is_resolution_constrained():
        return True

    def sampler(
        self,
        n=1,
        start_symbol: str = None,
        n_downsamples: int = None,
        depth_information: dict = None,
    ):
        if start_symbol is None:
            start_symbol = self.start()
        else:
            start_symbol = Nonterminal(start_symbol)

        if depth_information is None:
            depth_information = {}
        if n_downsamples is None:
            n_downsamples = self.n_downsamples
        return [
            f"{self._resolution_constrained_sampler(symbol=start_symbol, n_downsamples=n_downsamples, depth_information=depth_information)})"
            for _ in range(n)
        ]

    def _compute_depth_information_for_pre(self, tree: str) -> dict:
        depth_information = {nt: 0 for nt in self.nonterminals}
        q_nonterminals: Deque = deque()
        for split in tree.split(" "):
            if split == "":
                continue
            elif split[0] == "(":
                q_nonterminals.append(split[1:])
                depth_information[split[1:]] += 1
                continue
            while split[-1] == ")":
                nt = q_nonterminals.pop()
                depth_information[nt] -= 1
                split = split[:-1]
        return depth_information

    def _compute_depth_information(self, tree: str) -> tuple:
        split_tree = tree.split(" ")
        depth_information = [0] * len(split_tree)
        subtree_depth = [0] * len(split_tree)
        helper_subtree_depth = [0] * len(split_tree)
        helper_dict_depth_information = {nt: 0 for nt in self.nonterminals}
        helper_dict_subtree_depth: dict = {nt: deque() for nt in self.nonterminals}
        q_nonterminals: Deque = deque()
        for i, split in enumerate(split_tree):
            if split == "":
                continue
            elif split[0] == "(":
                nt = split[1:]
                q_nonterminals.append(nt)
                depth_information[i] = helper_dict_depth_information[nt] + 1
                helper_dict_depth_information[nt] += 1
                helper_dict_subtree_depth[nt].append(i)
                for j in helper_dict_subtree_depth[nt]:
                    subtree_depth[j] = max(subtree_depth[j], helper_subtree_depth[j] + 1)
                    helper_subtree_depth[j] += 1
                continue
            while split[-1] == ")":
                nt = q_nonterminals.pop()
                helper_dict_depth_information[nt] -= 1
                for j in helper_dict_subtree_depth[nt]:
                    helper_subtree_depth[j] -= 1
                _ = helper_dict_subtree_depth[nt].pop()
                split = split[:-1]
        return depth_information, subtree_depth

    def _compute_max_depth(self, tree: str, subtree_node: str) -> int:
        max_depth = 0
        depth_information = {nt: 0 for nt in self.nonterminals}
        q_nonterminals: Deque = deque()
        for split in tree.split(" "):
            if split == "":
                continue
            elif split[0] == "(":
                q_nonterminals.append(split[1:])
                depth_information[split[1:]] += 1
                if split[1:] == subtree_node and depth_information[split[1:]] > max_depth:
                    max_depth = depth_information[split[1:]]
                continue
            while split[-1] == ")":
                nt = q_nonterminals.pop()
                depth_information[nt] -= 1
                split = split[:-1]
        return max_depth

    @staticmethod
    def assign_downsamples(edge_list, paths, n_downsamples):
        if n_downsamples == 0:
            return [0] * len(edge_list)
        edge_list_to_downsamples = {e: 0 for e in edge_list}

        if max(paths.keys()) >= n_downsamples:
            for path in paths[n_downsamples]:
                for e in path:
                    edge_list_to_downsamples[e] = 1

        for k in reversed(sorted(paths.keys())):
            k_paths = paths[k]
            if len(k_paths) == 0 or k == n_downsamples:
                continue
            tmp_indices = list(range(len(k_paths)))
            np.random.shuffle(tmp_indices)
            for idx in tmp_indices:
                path = k_paths[idx]
                already_set_n_downsamples = sum(edge_list_to_downsamples[e] for e in path)
                if already_set_n_downsamples == n_downsamples:
                    continue
                _path = [e for e in path if edge_list_to_downsamples[e] == 0]

                _n_downsamples = n_downsamples - already_set_n_downsamples
                if len(_path) == 1:
                    edge_list_to_downsamples[path[0]] = _n_downsamples
                elif len(_path) < _n_downsamples:
                    indices = np.random.choice(
                        list(range(len(_path))),
                        size=n_downsamples // len(_path),
                        replace=False,
                    )
                    for i, e in enumerate(_path):
                        edge_list_to_downsamples[e] = (
                            n_downsamples // len(_path) + 1
                            if i in indices
                            else n_downsamples // len(_path)
                        )
                else:
                    indices = np.random.choice(
                        list(range(len(_path))),
                        size=_n_downsamples,
                        replace=False,
                    )
                    for i in indices:
                        edge_list_to_downsamples[_path[i]] = 1

        return [edge_list_to_downsamples[e] for e in edge_list]

    def _resolution_constrained_sampler(
        self, symbol=None, n_downsamples: int = 0, depth_information: dict = None
    ):
        if depth_information is None:
            depth_information = {}

        # init the sequence
        tree = "(" + str(symbol)

        lhs = str(symbol)
        if lhs in depth_information.keys():
            depth_information[lhs] += 1
        else:
            depth_information[lhs] = 1

        # collect possible productions from the starting symbol & filter if constraints are violated
        if lhs == self.downsampling_lhs:
            productions = [
                production
                for production in self.productions(lhs=symbol)
                if sum(str(x) == self.downsample_terminal for x in production.rhs())
                == n_downsamples
            ]
        elif (
            lhs in self.depth_constraints.keys()
            and depth_information[lhs] < self.depth_constraints[lhs]["min"]["number"]
        ):
            productions = [
                production
                for production in self.productions(lhs=symbol)
                if not (
                    len(production.rhs()) == 1
                    and str(production.rhs()[0])
                    in self.depth_constraints[lhs]["min"]["exclude_rhs"]
                )
            ]
        elif (
            lhs in self.depth_constraints.keys()
            and depth_information[lhs] >= self.depth_constraints[lhs]["max"]["number"]
        ):
            productions = [
                production
                for production in self.productions(lhs=symbol)
                if lhs
                not in [str(sym) for sym in production.rhs() if not isinstance(sym, str)]
            ]
        else:
            productions = self.productions(lhs=symbol)

        if len(productions) == 0:
            raise Exception(
                "There can be no word sampled! This is due to the grammar and/or constraints."
            )

        # sample
        production = choice(productions)
        n_downsamples_per_edge = []
        counter = 0
        for sym in production.rhs():
            if isinstance(sym, str):
                tree = tree + " " + sym
                if sym in self.terminal_to_graph_map.keys():
                    n_downsamples_per_edge = self.assign_downsamples(
                        self.terminal_to_graph_map[sym]["edge_list"],
                        self.terminal_to_graph_map[sym]["paths"],
                        n_downsamples,
                    )
            else:
                if counter < len(n_downsamples_per_edge):
                    _n_downsamples = n_downsamples_per_edge[counter]
                elif (
                    len(production.rhs()) == 1
                    and str(production.rhs()[0]) == self.downsampling_lhs
                ):
                    _n_downsamples = n_downsamples
                else:
                    _n_downsamples = 0
                tree = (
                    tree
                    + " "
                    + self._resolution_constrained_sampler(
                        sym,
                        n_downsamples=_n_downsamples,
                        depth_information=depth_information,
                    )
                    + ")"
                )
                counter += 1

        depth_information[lhs] -= 1
        return tree

    def mutate(
        self, parent: str, subtree_index: int, subtree_node: str, patience: int = 50
    ) -> str:
        # chop out subtree
        pre, _, post = self.remove_subtree(parent, subtree_index)
        _patience = patience
        while _patience > 0:
            # only sample subtree -> avoids full sampling of large parse trees
            depth_information = self._compute_depth_information_for_pre(pre)
            new_subtree = self.sampler(
                1, start_symbol=subtree_node, depth_information=depth_information
            )[0]
            child = pre + new_subtree + post
            if parent != child:  # ensure that parent is really mutated
                break
            _patience -= 1
        child = self._remove_empty_spaces(child)
        return child

    def crossover(
        self,
        parent1: str,
        parent2: str,
        patience: int = 50,
        return_crossover_subtrees: bool = False,
    ):
        # randomly swap subtrees in two trees
        # if no suitiable subtree exists then return False
        subtree_node, subtree_index = self.rand_subtree(parent1)
        # chop out subtree
        pre, sub, post = self.remove_subtree(parent1, subtree_index)
        head_node_depth = self._compute_depth_information_for_pre(pre)[subtree_node] + 1
        sub_depth = self._compute_max_depth(sub, subtree_node)
        _patience = patience
        while _patience > 0:
            # sample subtree from donor
            donor_subtree_index = self._rand_subtree_fixed_head(
                parent2, subtree_node, head_node_depth, sub_depth=sub_depth
            )
            # if no subtrees with right head node return False
            if not donor_subtree_index:
                _patience -= 1
            else:
                donor_pre, donor_sub, donor_post = self.remove_subtree(
                    parent2, donor_subtree_index
                )
                # return the two new tree
                child1 = pre + donor_sub + post
                child2 = donor_pre + sub + donor_post

                child1 = self._remove_empty_spaces(child1)
                child2 = self._remove_empty_spaces(child2)

                if return_crossover_subtrees:
                    return (
                        child1,
                        child2,
                        (pre, sub, post),
                        (donor_pre, donor_sub, donor_post),
                    )

                return child1, child2

        return False, False

    def _rand_subtree_fixed_head(
        self,
        tree: str,
        head_node: str,
        head_node_depth: int = 0,
        sub_depth: int = 0,
    ) -> int:
        # helper function to choose a random subtree from a given tree with a specific head node
        # if no such subtree then return False, otherwise return the index of the subtree

        # single pass through tree (stored as string) to look for the location of swappable_non_terminmals
        if head_node in self.depth_constraints:
            depth_information, subtree_depth = self._compute_depth_information(tree)
            split_tree = tree.split(" ")
            swappable_indices = [
                i
                for i in range(len(split_tree))
                if split_tree[i][1:] == head_node
                and head_node_depth - 1 + subtree_depth[i]
                <= self.depth_constraints[head_node]
                and depth_information[i] - 1 + sub_depth
                <= self.depth_constraints[head_node]
            ]
        else:
            swappable_indices = None
        return super().rand_subtree_fixed_head(
            tree=tree, head_node=head_node, swappable_indices=swappable_indices
        )
