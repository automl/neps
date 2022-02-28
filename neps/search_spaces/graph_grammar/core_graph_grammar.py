from __future__ import annotations

import collections
import inspect
import queue
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from typing import Deque

import networkx as nx
import numpy as np

from .cfg import Grammar
from .graph import Graph
from .primitives import AbstractPrimitive


class CoreGraphGrammar(Graph):
    def __init__(
        self,
        grammars: list[Grammar],
        terminal_to_op_names: dict,
        terminal_to_graph_repr: dict,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = None,
        identity_op: list = None,
        name: str = None,
        scope: str = None,
    ):
        super().__init__(name, scope)

        self.grammars = grammars
        self.terminal_to_op_names = terminal_to_op_names
        self.terminal_to_graph_repr = terminal_to_graph_repr
        self.edge_attr = edge_attr
        self.edge_label = edge_label

        self.zero_op = zero_op if zero_op is not None else []
        self.identity_op = identity_op if identity_op is not None else []

        self.terminal_to_graph_nodes: dict = {}

    def get_grammars(self) -> list[Grammar]:
        return self.grammars

    def clear_graph(self):
        while len(self.nodes()) != 0:
            self.remove_node(list(self.nodes())[0])

    @staticmethod
    def _check_graph(graph: nx.DiGraph):
        if len(graph) == 0 or graph.number_of_edges() == 0:
            raise ValueError("Invalid DAG")

    @abstractmethod
    def get_graphs(self):
        raise NotImplementedError

    def prune_tree(
        self,
        tree: nx.DiGraph,
        terminal_to_torch_map_keys: collections.abc.KeysView,
        node_label: str = "op_name",
    ) -> nx.DiGraph:
        """Prunes unnecessary parts of parse tree, i.e., only one child

        Args:
            tree (nx.DiGraph): Parse tree

        Returns:
            nx.DiGraph: Pruned parse tree
        """

        def dfs(visited: set, tree: nx.DiGraph, node: int) -> nx.DiGraph:
            if node not in visited:
                visited.add(node)

                i = 0
                while i < len(tree.nodes[node]["children"]):
                    former_len = len(tree.nodes[node]["children"])
                    child = tree.nodes[node]["children"][i]
                    tree = dfs(
                        visited,
                        tree,
                        child,
                    )
                    if former_len == len(tree.nodes[node]["children"]):
                        i += 1

                if len(tree.nodes[node]["children"]) == 1:
                    predecessor = list(tree.pred[node])
                    if len(predecessor) > 0:
                        tree.add_edge(predecessor[0], tree.nodes[node]["children"][0])
                        old_children = tree.nodes[predecessor[0]]["children"]
                        idx = [i for i, c in enumerate(old_children) if c == node][0]
                        tree.nodes[predecessor[0]]["children"] = (
                            old_children[: idx + 1]
                            + [tree.nodes[node]["children"][0]]
                            + old_children[idx + 1 :]
                        )
                        tree.nodes[predecessor[0]]["children"].remove(node)

                    tree.remove_node(node)
                elif (
                    tree.nodes[node]["terminal"]
                    and tree.nodes[node][node_label] not in terminal_to_torch_map_keys
                ):
                    predecessor = list(tree.pred[node])[0]
                    tree.nodes[predecessor]["children"].remove(node)
                    tree.remove_node(node)
            return tree

        return dfs(set(), tree, self._find_root(tree))

    @staticmethod
    def _dfs_preorder_nodes(G: nx.DiGraph, source: str = None) -> list[int]:
        """Generates nodes in DFS pre-ordering starting at source.
        Note that after pruning we cannot reconstruct the associated string tree!

        Args:
            G (nx.DiGraph): NetworkX DAG
            source (str, optional): Starting node for DFS. Defaults to None.

        Returns:
            generator: List of nodes in a DFS pre-ordering.
        """
        edges = nx.dfs_labeled_edges(G, source=source)
        return list(v for _, v, d in edges if d == "forward")

    @staticmethod
    def _find_leafnodes(G):
        leafnode = []
        for i in G.nodes:
            head = []
            if nx.descendants(G, i) == set():  # find all leaf nodes
                for a in nx.ancestors(G, i):  # get all ancestors for leaf node
                    if (
                        nx.ancestors(G, a) == set()
                    ):  # Determine if ancestor is a head node
                        head.append(a)
            if len(head) == 1:  # if this leaf had only one head then append to leafnode
                leafnode.append(i)
        return leafnode

    @staticmethod
    def _get_neighbors_from_parse_tree(tree: nx.DiGraph, node: int) -> list[int]:
        return tree.nodes[node]["children"]

    @staticmethod
    def _find_root(G):
        return [n for n, d in G.in_degree() if d == 0][0]

    @staticmethod
    def _relabel_nodes(G: nx.DiGraph, mapping: dict) -> nx.DiGraph:
        """Relabels the nodes and adjusts children list accordingly.

        Args:
            G (nx.DiGraph): graph to relabel
            mapping (dict): node mapping

        Returns:
            nx.DiGraph: relabeled graph (copied)
        """
        # recreation of graph is faster
        tree_relabeled = nx.DiGraph()
        tree_relabeled.add_nodes_from(
            [
                (
                    mapping[n[0]],
                    {
                        k: v if k != "children" else [mapping[_n] for _n in v]
                        for k, v in n[1].items()
                    },
                )
                for n in G.nodes(data=True)
            ]
        )
        tree_relabeled.add_edges_from([(mapping[e[0]], mapping[e[1]]) for e in G.edges()])
        return tree_relabeled

    def assemble_trees(
        self,
        base_tree: str | nx.DiGraph,
        motif_trees: list[str] | list[nx.DiGraph],
        base_to_motif_map: dict = None,
        node_label: str = "op_name",
    ) -> str | nx.DiGraph:
        """Assembles the base parse tree with the motif parse trees

        Args:
            base_tree (nx.DiGraph): Base parse tree
            motif_trees (List[nx.DiGraph]): List of motif parse trees
            node_label (str, optional): node label key. Defaults to "op_name".

        Returns:
            nx.DiGraph: Assembled parse tree
        """
        if not all([isinstance(base_tree, type(tree)) for tree in motif_trees]):
            raise ValueError("All trees must be of the same type!")
        if isinstance(base_tree, str):
            ensembled_tree_string = base_tree
            if base_to_motif_map is None:
                raise NotImplementedError

            for motif, idx in base_to_motif_map.items():
                if motif in ensembled_tree_string:
                    replacement = motif_trees[idx]
                    ensembled_tree_string = ensembled_tree_string.replace(
                        motif, replacement
                    )
            return ensembled_tree_string
        elif isinstance(base_tree, nx.DiGraph):
            leafnodes = self._find_leafnodes(base_tree)
            root_nodes = [self._find_root(G) for G in motif_trees]
            root_op_names = np.array(
                [
                    motif_tree.nodes[root_node][node_label]
                    for motif_tree, root_node in zip(motif_trees, root_nodes)
                ]
            )
            largest_node_number = max(base_tree.nodes())
            # ensembled_tree = base_tree.copy()
            # recreation is slightly faster
            ensembled_tree: nx.DiGraph = nx.DiGraph()
            ensembled_tree.add_nodes_from(base_tree.nodes(data=True))
            ensembled_tree.add_edges_from(base_tree.edges())
            for leafnode in leafnodes:
                idx = np.where(base_tree.nodes[leafnode][node_label] == root_op_names)[0]
                if len(idx) == 0:
                    continue
                if len(idx) > 1:
                    raise ValueError(
                        "More than two similar terminal/start symbols are not supported!"
                    )

                tree = motif_trees[idx[0]]
                # generate mapping
                mapping = {
                    n: n_new
                    for n, n_new in zip(
                        tree.nodes(),
                        range(
                            largest_node_number + 1,
                            largest_node_number + 1 + len(tree),
                        ),
                    )
                }
                largest_node_number = largest_node_number + 1 + len(tree)
                tree_relabeled = self._relabel_nodes(G=tree, mapping=mapping)

                # compose trees
                predecessor_in_base_tree = list(ensembled_tree.pred[leafnode])[0]
                motif_tree_root_node = self._find_root(tree_relabeled)
                successors_in_motif_tree = tree_relabeled.nodes[motif_tree_root_node][
                    "children"
                ]

                # delete unnecessary edges
                ensembled_tree.remove_node(leafnode)
                tree_relabeled.remove_node(motif_tree_root_node)
                # add new edges
                tree_relabeled.add_node(predecessor_in_base_tree)
                for n in successors_in_motif_tree:
                    tree_relabeled.add_edge(predecessor_in_base_tree, n)

                ensembled_tree.update(
                    edges=tree_relabeled.edges(data=True),
                    nodes=tree_relabeled.nodes(data=True),
                )

                idx = np.where(
                    np.array(ensembled_tree.nodes[predecessor_in_base_tree]["children"])
                    == leafnode
                )[0][0]
                old_children = ensembled_tree.nodes[predecessor_in_base_tree]["children"]
                ensembled_tree.nodes[predecessor_in_base_tree]["children"] = (
                    old_children[: idx + 1]
                    + successors_in_motif_tree
                    + old_children[idx + 1 :]
                )
                ensembled_tree.nodes[predecessor_in_base_tree]["children"].remove(
                    leafnode
                )
            return ensembled_tree
        else:
            raise NotImplementedError(
                f"Assembling of trees of type {type(base_tree)} is not supported!"
            )

    def build_graph_from_tree(
        self,
        tree: nx.DiGraph,
        terminal_to_torch_map: dict,
        node_label: str = "op_name",
        flatten_graph: bool = True,
        return_cell: bool = False,
    ) -> None | Graph:
        """Builds the computational graph from a parse tree.

        Args:
            tree (nx.DiGraph): parse tree.
            terminal_to_torch_map (dict): Mapping from terminal symbols to primitives or topologies.
            node_label (str, optional): Key to access terminal symbol. Defaults to "op_name".
            return_cell (bool, optional): Whether to return a cell. Is only needed if cell is repeated multiple times.
            Defaults to False.

        Returns:
            Tuple[Union[None, Graph]]: computational graph (self) or cell.
        """

        def _build_graph_from_tree(
            visited: set,
            tree: nx.DiGraph,
            node: int,
            terminal_to_torch_map: dict,
            node_label: str,
            is_primitive: bool = False,
        ):
            """Recursive DFS-esque function to build computational graph from parse tree

            Args:
                visited (set): set of visited nodes.
                tree (nx.DiGraph): parse tree.
                node (int): node index.
                terminal_to_torch_map (dict): mapping from terminal symbols to primitives or topologies.
                node_label (str): key to access operation name

            Raises:
                Exception: primitive or topology is unknown, i.e., it is probably missing in the terminal to
                torch mapping
                Exception: leftmost children can only be primitive, topology or have one child

            Returns:
                [type]: computational graph.
            """
            if node not in visited:
                subgraphs = []
                primitive_hps = []
                if len(tree.out_edges(node)) == 0:
                    if is_primitive:
                        return tree.nodes[node][node_label]
                    else:
                        if (
                            tree.nodes[node][node_label]
                            not in terminal_to_torch_map.keys()
                        ):
                            raise Exception(
                                f"Unknown primitive or topology: {tree.nodes[node][node_label]}"
                            )
                        return deepcopy(
                            terminal_to_torch_map[tree.nodes[node][node_label]]
                        )
                if len(tree.out_edges(node)) == 1:
                    return _build_graph_from_tree(
                        visited,
                        tree,
                        list(tree.neighbors(node))[0],
                        terminal_to_torch_map,
                        node_label,
                        is_primitive,
                    )
                # for idx, neighbor in enumerate(tree.neighbors(node)):
                for idx, neighbor in enumerate(
                    self._get_neighbors_from_parse_tree(tree, node)
                ):
                    if idx == 0:  # topology or primitive
                        n = neighbor
                        while not tree.nodes[n]["terminal"]:
                            if len(tree.out_edges(n)) != 1:
                                raise Exception(
                                    "Leftmost Child can only be primitive, topology or recursively have one child!"
                                )
                            n = next(tree.neighbors(n))
                        if is_primitive:
                            primitive_hp_key = tree.nodes[n][node_label]
                            primitive_hp_dict = {primitive_hp_key: None}
                            is_primitive_op = True
                        else:
                            if (
                                tree.nodes[n][node_label]
                                not in terminal_to_torch_map.keys()
                            ):
                                raise Exception(
                                    f"Unknown primitive or topology: {tree.nodes[n][node_label]}"
                                )
                            graph_el = terminal_to_torch_map[tree.nodes[n][node_label]]
                            is_primitive_op = issubclass(
                                graph_el.func
                                if isinstance(graph_el, partial)
                                else graph_el,
                                AbstractPrimitive,
                            )
                    elif not tree.nodes[neighbor][
                        "terminal"
                    ]:  # exclude '[' ']' ... symbols
                        if is_primitive:
                            primitive_hp_dict[primitive_hp_key] = _build_graph_from_tree(
                                visited,
                                tree,
                                neighbor,
                                terminal_to_torch_map,
                                node_label,
                                is_primitive_op,
                            )
                        elif is_primitive_op:
                            primitive_hps.append(
                                _build_graph_from_tree(
                                    visited,
                                    tree,
                                    neighbor,
                                    terminal_to_torch_map,
                                    node_label,
                                    is_primitive_op,
                                )
                            )
                        else:
                            subgraphs.append(
                                _build_graph_from_tree(
                                    visited,
                                    tree,
                                    neighbor,
                                    terminal_to_torch_map,
                                    node_label,
                                    is_primitive_op,
                                )
                            )
                    elif (
                        tree.nodes[neighbor][node_label] in terminal_to_torch_map.keys()
                    ):  # exclude '[' ']' ... symbols
                        # TODO check if there is a potential bug here?
                        subgraphs.append(
                            deepcopy(
                                terminal_to_torch_map[tree.nodes[neighbor][node_label]]
                            )
                        )

                if is_primitive:
                    return primitive_hp_dict
                elif is_primitive_op:
                    return dict(
                        collections.ChainMap(*([{"op": graph_el}] + primitive_hps))
                    )
                else:
                    return graph_el(*subgraphs)

        def _flatten_graph(
            graph,
            flattened_graph,
            start_node: int = None,
            end_node: int = None,
        ):
            nodes: dict = {}
            for u, v, data in graph.edges(data=True):
                if u in nodes.keys():
                    _u = nodes[u]
                else:
                    _u = (
                        1
                        if len(flattened_graph.nodes.keys()) == 0
                        else max(flattened_graph.nodes.keys()) + 1
                    )
                    _u = (
                        start_node
                        if graph.in_degree(u) == 0 and start_node is not None
                        else _u
                    )
                    nodes[u] = _u
                    if _u not in flattened_graph.nodes.keys():
                        flattened_graph.add_node(_u)

                if v in nodes.keys():
                    _v = nodes[v]
                else:
                    _v = max(flattened_graph.nodes.keys()) + 1
                    _v = (
                        end_node
                        if graph.out_degree(v) == 0 and end_node is not None
                        else _v
                    )
                    nodes[v] = _v
                    if _v not in flattened_graph.nodes.keys():
                        flattened_graph.add_node(_v)

                if isinstance(data["op"], Graph):
                    flattened_graph = _flatten_graph(
                        data["op"], flattened_graph, start_node=_u, end_node=_v
                    )
                else:
                    flattened_graph.add_edge(_u, _v)
                    flattened_graph.edges[_u, _v].update(data)

            return flattened_graph

        root_node = self._find_root(tree)
        graph = _build_graph_from_tree(
            set(), tree, root_node, terminal_to_torch_map, node_label
        )
        self._check_graph(graph)
        if return_cell:
            cell = (
                _flatten_graph(graph, flattened_graph=Graph()) if flatten_graph else graph
            )
            return cell
        else:
            if flatten_graph:
                _flatten_graph(graph, flattened_graph=self)
            else:
                self.add_edge(0, 1)
                self.edges[0, 1].set("op", graph)
            return None

    def to_graph_repr(self, graph: Graph, edge_attr: bool) -> nx.DiGraph:
        """Transforms NASLib-esque graph to NetworkX graph.

        Args:
            graph (Graph): NASLib-esque graph.
            edge_attr (bool): Transform to edge attribution or node attribution.

        Returns:
            nx.DiGraph: edge- or node-attributed representation of computational graph.
        """
        if edge_attr:
            g = nx.DiGraph()
            g.add_nodes_from(graph.nodes())
            for u, v in graph.edges():
                if isinstance(graph.edges[u, v]["op"], Graph):
                    g.add_edge(u, v, op_name=graph.edges[u, v]["op"].name)
                else:
                    g.add_edge(
                        u, v, **{self.edge_label: graph.edges[u, v][self.edge_label]}
                    )
            g.graph_type = "edge_attr"
        else:
            g = nx.DiGraph()
            src = [n for n in graph.nodes() if graph.in_degree(n) == 0][0]
            tgt = [n for n in graph.nodes() if graph.out_degree(n) == 0][0]
            nof_edges = graph.size()
            g.add_nodes_from(
                [
                    (0, {self.edge_label: "input"}),
                    (nof_edges + 1, {self.edge_label: "output"}),
                ]
            )
            node_counter = 1
            open_edge: dict = {}
            for node in nx.topological_sort(graph):
                for edge in graph.out_edges(node):
                    g.add_node(
                        node_counter,
                        **{self.edge_label: graph.edges[edge][self.edge_label]},
                    )

                    u, v = edge
                    if u == src:  # special case for input node
                        g.add_edge(0, node_counter)
                    if v == tgt:  # special case of output node
                        g.add_edge(node_counter, nof_edges + 1)
                    if (
                        u in open_edge.keys()
                    ):  # add edge between already seen nodes and new node
                        for node_count in open_edge[u]:
                            g.add_edge(node_count, node_counter)

                    if v in open_edge.keys():
                        open_edge[v].append(node_counter)
                    else:
                        open_edge[v] = [node_counter]
                    node_counter += 1
            g.graph_type = "node_attr"

        self._check_graph(g)

        return g

    @staticmethod
    def from_stringTree_to_nxTree(
        string_tree: str, grammar: Grammar, sym_name: str = "op_name"
    ) -> nx.DiGraph:
        """Transforms a parse tree from string representation to NetworkX representation.

        Args:
            string_tree (str): parse tree.
            grammar (Grammar): context-free grammar which generated the parse tree in string represenation.
            sym_name (str, optional): Key to save the terminal symbols. Defaults to "op_name".

        Returns:
            nx.DiGraph: parse tree as NetworkX representation.
        """

        def skip_char(char: str) -> bool:
            if char in [" ", "\t", "\n"]:
                return True
            # special case: "(" is (part of) a terminal
            if (
                i != 0
                and char == "("
                and string_tree[i - 1] == " "
                and string_tree[i + 1] == " "
            ):
                return False
            if char == "(":
                return True
            return False

        def find_longest_match(
            i: int, string_tree: str, symbols: list[str], max_match: int
        ) -> int:
            # search for longest matching symbol and add it
            # assumes that the longest match is the true match
            j = min(i + max_match, len(string_tree) - 1)
            while j > i and j < len(string_tree):
                if string_tree[i:j] in symbols:
                    break
                j -= 1
            if j == i:
                raise Exception(f"Terminal or nonterminal at position {i} does not exist")
            return j

        if isinstance(grammar, list) and len(grammar) > 1:
            full_grammar = deepcopy(grammar[0])
            rules = full_grammar.productions()
            nonterminals = full_grammar.nonterminals
            terminals = full_grammar.terminals
            for g in grammar[1:]:
                rules.extend(g.productions())
                nonterminals.extend(g.nonterminals)
                terminals.extend(g.terminals)
            grammar = full_grammar
            raise NotImplementedError("TODO check implementation")

        symbols = grammar.nonterminals + grammar.terminals
        max_match = max(map(len, symbols))
        find_longest_match_func = partial(
            find_longest_match,
            string_tree=string_tree,
            symbols=symbols,
            max_match=max_match,
        )

        G = nx.DiGraph()
        q: queue.LifoQueue = queue.LifoQueue()
        q_children: queue.LifoQueue = queue.LifoQueue()
        node_number = 0
        i = 0
        while i < len(string_tree):
            char = string_tree[i]
            if skip_char(char):
                pass
            elif char == ")" and not string_tree[i - 1] == " ":
                # closing symbol of production
                _node_number = q.get(block=False)
                _node_children = q_children.get(block=False)
                G.nodes[_node_number]["children"] = _node_children
            else:
                j = find_longest_match_func(i)
                sym = string_tree[i:j]
                i = j - 1
                node_number += 1
                G.add_node(
                    node_number,
                    **{
                        sym_name: sym,
                        "terminal": sym in grammar.terminals,
                        "children": [],
                    },
                )
                if not q.empty():
                    G.add_edge(q.queue[-1], node_number)
                    q_children.queue[-1].append(node_number)
                if sym in grammar.nonterminals:
                    q.put(node_number)
                    q_children.put([])
            i += 1

        if len(q.queue) != 0:
            raise Exception("Invalid string_tree")
        return G

    def from_nxTree_to_stringTree(
        self, nxTree: nx.DiGraph, node_label: str = "op_name"
    ) -> str:
        """Transforms parse tree represented as NetworkX DAG to string representation.

        Args:
            nxTree (nx.DiGraph): parse tree.
            node_label (str, optional): key to access operation names. Defaults to "op_name".

        Returns:
            str: parse tree represented as string.
        """

        def dfs(visited, graph, node):
            if node not in visited:
                visited.add(node)
                if graph.nodes[node]["terminal"]:
                    return f"{graph.nodes[node][node_label]}"
                tmp_str = f"{f'({graph.nodes[node][node_label]}'}" + " "
                # for neighbor in graph.neighbors(node):
                for neighbor in self._get_neighbors_from_parse_tree(graph, node):
                    tmp_str += dfs(visited, graph, neighbor) + " "
                tmp_str = tmp_str[:-1] + ")"
                return tmp_str
            return ""

        return dfs(set(), nxTree, node=self._find_root(nxTree))

    def update_op_names(self):
        # update op names
        for u, v in self.edges():
            try:
                self.edges[u, v].update({"op_name": self.edges[u, v]["op"].get_op_name})
            except Exception:
                self.edges[u, v].update({"op_name": self.edges[u, v]["op"].name})

    def from_stringTree_to_graph_repr(
        self,
        string_tree: str,
        grammar: Grammar,
        terminal_to_graph_edges: dict,
        valid_terminals: collections.abc.KeysView,
        edge_attr: bool = True,
        sym_name: str = "op_name",
        prune: bool = True,
        add_subtree_map: bool = False,
        return_subgraph_dict: bool = False,
    ) -> nx.DiGraph | tuple[nx.DiGraph, collections.OrderedDict]:
        """Generates graph from parse tree in string representation.
        Note that we ignore primitive HPs!

        Args:
            string_tree (str): parse tree.
            grammar (Grammar): underlying grammar.
            terminal_to_graph_edges (dict): map from terminal symbol to topology or primitive operation.
            valid_terminals (list): list of valid terminal symbols.
            edge_attr (bool, optional): Shoud graph be edge attributed (True) or node attributed (False). Defaults to True.
            sym_name (str, optional): Attribute name of operation. Defaults to "op_name".
            prune (bool, optional): Prune graph, e.g., None operations etc. Defaults to True.
            add_subtree_map (bool, optional): Add attribute indicating to which subtrees of
                the parse tree the specific part belongs to. Can only be true if you set prune=False!
                TODO: Check if we really need this constraint or can also allow pruning. Defaults to False.
            return_subgraph_dict (bool, optional): Additionally returns an hierarchical dictionary
                containing all subgraphs. Defaults to False.
                TODO: check if edge attr also works.

        Returns:
            nx.DiGraph: [description]
        """
        if prune:
            add_subtree_map = False

        def to_node_attributed_edge_list(
            edge_list: list[tuple],
        ) -> tuple[list[tuple[int, int]], dict]:
            node_offset = 2
            edge_to_node_map = {e: i + node_offset for i, e in enumerate(edge_list)}
            first_nodes = {e[0] for e in edge_list}
            second_nodes = {e[1] for e in edge_list}
            (src,) = first_nodes - second_nodes
            (tgt,) = second_nodes - first_nodes
            node_list = []
            for e in edge_list:
                ni = edge_to_node_map[e]
                u, v = e
                if u == src:
                    node_list.append((0, ni))
                if v == tgt:
                    node_list.append((ni, 1))

                for e_ in filter(
                    lambda e: (e[1] == u), edge_list  # pylint: disable=W0640
                ):
                    node_list.append((edge_to_node_map[e_], ni))

            return node_list, edge_to_node_map

        def skip_char(char: str) -> bool:
            return True if char in [" ", "\t", "\n", "[", "]"] else False

        G = nx.DiGraph()
        if add_subtree_map:
            q_nonterminals: Deque = collections.deque()
        if return_subgraph_dict:
            q_subtrees: Deque = collections.deque()
            q_subgraphs: Deque = collections.deque()
            subgraphs_dict = collections.OrderedDict()
        if edge_attr:
            node_offset = 0
            q_el: Deque = collections.deque()  # edge-attr
            terminal_to_graph = terminal_to_graph_edges
        else:  # node-attributed
            G.add_node(0, **{sym_name: "input"})
            G.add_node(1, **{sym_name: "output"})
            node_offset = 2
            if bool(self.terminal_to_graph_nodes):
                terminal_to_graph_nodes = self.terminal_to_graph_nodes
            else:
                terminal_to_graph_nodes = {
                    k: to_node_attributed_edge_list(edge_list)
                    for k, edge_list in terminal_to_graph_edges.items()
                }
                self.terminal_to_graph_nodes = terminal_to_graph_nodes
            terminal_to_graph = {k: v[0] for k, v in terminal_to_graph_nodes.items()}
            q_el = collections.deque()  # node-attr

        # pre-compute stuff
        begin_end_nodes = {}
        for sym, g in terminal_to_graph.items():
            first_nodes = {e[0] for e in g}
            second_nodes = {e[1] for e in g}
            (begin_node,) = first_nodes - second_nodes
            (end_node,) = second_nodes - first_nodes
            begin_end_nodes[sym] = (begin_node, end_node)

        for split_idx, sym in enumerate(string_tree.split(" ")):
            if sym == "":
                continue
            if return_subgraph_dict:
                new_sym = True
                sym_copy = sym[:]
            if sym[0] == "(":
                sym = sym[1:]
            if sym[-1] == ")":
                if add_subtree_map:
                    for _ in range(sym.count(")")):
                        q_nonterminals.pop()
                if return_subgraph_dict:
                    new_sym = False
                while sym[-1] == ")" and sym not in valid_terminals:
                    sym = sym[:-1]

            if return_subgraph_dict and new_sym:
                if sym in grammar.nonterminals:
                    # need dict as a graph can have multiple subgraphs
                    q_subtrees.append(sym_copy[:])
                else:
                    q_subtrees[-1] += f" {sym_copy}"

            if len(sym) == 1 and skip_char(sym[0]):
                continue

            if add_subtree_map and sym in grammar.nonterminals:
                q_nonterminals.append((sym, split_idx))
            elif sym in valid_terminals:  # terminal symbol
                if sym in terminal_to_graph_edges:
                    if len(q_el) == 0:
                        if edge_attr:
                            edges = [
                                tuple(t + node_offset for t in e)
                                for e in terminal_to_graph_edges[sym]
                            ]
                        else:  # node-attr
                            edges = [
                                tuple(t for t in e)
                                for e in terminal_to_graph_nodes[sym][0]
                            ]
                            nodes = [
                                terminal_to_graph_nodes[sym][1][e]
                                for e in terminal_to_graph_edges[sym]
                            ]
                        if add_subtree_map:
                            subtrees = []
                        first_nodes = {e[0] for e in edges}
                        second_nodes = {e[1] for e in edges}
                        (src_node,) = first_nodes - second_nodes
                        (sink_node,) = second_nodes - first_nodes
                    else:
                        begin_node, end_node = begin_end_nodes[sym]
                        el = q_el.pop()
                        if edge_attr:
                            u, v = el
                            if add_subtree_map:
                                subtrees = G[u][v]["subtrees"]
                            G.remove_edge(u, v)
                            edges = [
                                tuple(
                                    u
                                    if t == begin_node
                                    else v
                                    if t == end_node
                                    else t + node_offset
                                    for t in e
                                )
                                for e in terminal_to_graph_edges[sym]
                            ]
                        else:  # node-attr
                            n = el
                            if add_subtree_map:
                                subtrees = G.nodes[n]["subtrees"]
                            in_nodes = list(G.predecessors(n))
                            out_nodes = list(G.successors(n))
                            G.remove_node(n)
                            edges = []
                            for e in terminal_to_graph_nodes[sym][0]:
                                if not (e[0] == begin_node or e[1] == end_node):
                                    edges.append((e[0] + node_offset, e[1] + node_offset))
                                elif e[0] == begin_node:
                                    for nin in in_nodes:
                                        edges.append((nin, e[1] + node_offset))
                                elif e[1] == end_node:
                                    for nout in out_nodes:
                                        edges.append((e[0] + node_offset, nout))
                            nodes = [
                                terminal_to_graph_nodes[sym][1][e] + node_offset
                                for e in terminal_to_graph_edges[sym]
                            ]

                    G.add_edges_from(edges)

                    if return_subgraph_dict:
                        subgraph = nx.DiGraph()
                        subgraph.add_edges_from(edges)
                        q_subgraphs.append(
                            {
                                "graph": subgraph,
                                "atoms": collections.OrderedDict(
                                    (atom, None)
                                    for atom in (edges if edge_attr else nodes)
                                ),
                            }
                        )

                    if add_subtree_map:
                        if edge_attr:
                            subtrees.append(q_nonterminals[-1])
                            for u, v in edges:
                                G[u][v]["subtrees"] = subtrees.copy()
                        else:  # node-attr
                            subtrees.append(q_nonterminals[-1])
                            for n in nodes:
                                G.nodes[n]["subtrees"] = subtrees.copy()

                    q_el.extend(reversed(edges if edge_attr else nodes))
                    if edge_attr:
                        node_offset += max(max(terminal_to_graph_edges[sym]))
                    else:
                        node_offset += max(terminal_to_graph_nodes[sym][1].values())
                else:  # primitive operations
                    el = q_el.pop()
                    if edge_attr:
                        u, v = el
                        if prune and sym in self.zero_op:
                            G.remove_edge(u, v)
                            if return_subgraph_dict:
                                q_subgraphs[-1]["graph"].remove_edge(u, v)
                                del q_subgraphs[-1]["atoms"][(u, v)]
                        else:
                            G[u][v][sym_name] = sym
                            if return_subgraph_dict:
                                q_subgraphs[-1]["graph"][u][v][sym_name] = sym
                            if add_subtree_map:
                                G[u][v]["subtrees"].append(q_nonterminals[-1])
                                q_nonterminals.pop()
                    else:  # node-attr
                        n = el
                        if prune and sym in self.zero_op:
                            G.remove_node(n)
                            if return_subgraph_dict:
                                q_subgraphs[-1]["graph"].remove_node(n)
                                del q_subgraphs[-1]["atoms"][n]
                        elif prune and sym in self.identity_op:
                            G.add_edges_from(
                                [
                                    (n_in, n_out)
                                    for n_in in G.predecessors(n)
                                    for n_out in G.successors(n)
                                ]
                            )
                            G.remove_node(n)
                            if return_subgraph_dict:
                                q_subgraphs[-1]["graph"].add_edges_from(
                                    [
                                        (n_in, n_out)
                                        for n_in in q_subgraphs[-1]["graph"].predecessors(
                                            n
                                        )
                                        for n_out in q_subgraphs[-1]["graph"].successors(
                                            n
                                        )
                                    ]
                                )
                                q_subgraphs[-1]["graph"].remove_node(n)
                                del q_subgraphs[-1]["atoms"][n]
                        else:
                            G.nodes[n][sym_name] = sym
                            if return_subgraph_dict:
                                q_subgraphs[-1]["graph"].nodes[n][sym_name] = sym
                                q_subgraphs[-1]["atoms"][
                                    next(
                                        filter(
                                            lambda x: x[1] is None,
                                            q_subgraphs[-1]["atoms"].items(),
                                        )
                                    )[0]
                                ] = sym
                            if add_subtree_map:
                                G.nodes[n]["subtrees"].append(q_nonterminals[-1])
                                q_nonterminals.pop()
            if return_subgraph_dict and sym_copy[-1] == ")":
                q_subtrees[-1] += f" {sym_copy}"
                for _ in range(sym_copy.count(")")):
                    subtree_identifier = q_subtrees.pop()
                    if len(q_subtrees) > 0:
                        q_subtrees[-1] += f" {subtree_identifier}"
                    if len(q_subtrees) == len(q_subgraphs) - 1:
                        difference = subtree_identifier.count(
                            "("
                        ) - subtree_identifier.count(")")
                        if difference < 0:
                            subtree_identifier = subtree_identifier[:difference]
                        subgraph_dict = q_subgraphs.pop()
                        subgraph = subgraph_dict["graph"]
                        atoms = subgraph_dict["atoms"]
                        if len(q_subtrees) > 0:
                            # subtree_identifier is subgraph graph at [-1]
                            # (and sub-...-subgraph currently in q_subgraphs)
                            q_subgraphs[-1]["atoms"][
                                next(
                                    filter(
                                        lambda x: x[1] is None,
                                        q_subgraphs[-1]["atoms"].items(),
                                    )
                                )[0]
                            ] = subtree_identifier

                        for atom in filter(lambda x: x[1] is not None, atoms.items()):
                            if edge_attr:
                                subgraph[atom[0][0]][atom[0][1]][sym_name] = atom[1]
                            else:  # node-attr
                                subgraph.nodes[atom[0]][sym_name] = atom[1]

                        if not edge_attr:  # node-attr
                            # ensure there is actually one input and output node
                            first_nodes = {e[0] for e in subgraph.edges}
                            second_nodes = {e[1] for e in subgraph.edges}
                            new_src_node = max(subgraph.nodes) + 1
                            src_nodes = first_nodes - second_nodes
                            subgraph.add_edges_from(
                                [
                                    (new_src_node, successor)
                                    for src_node in src_nodes
                                    for successor in subgraph.successors(src_node)
                                ]
                            )
                            subgraph.add_node(new_src_node, **{sym_name: "input"})
                            subgraph.remove_nodes_from(src_nodes)
                            new_sink_node = max(subgraph.nodes) + 1
                            sink_nodes = second_nodes - first_nodes
                            subgraph.add_edges_from(
                                [
                                    (predecessor, new_sink_node)
                                    for sink_node in sink_nodes
                                    for predecessor in subgraph.predecessors(sink_node)
                                ]
                            )
                            subgraph.add_node(new_sink_node, **{sym_name: "output"})
                            subgraph.remove_nodes_from(sink_nodes)
                        subgraphs_dict[subtree_identifier] = subgraph

        if len(q_el) != 0:
            raise Exception("Invalid string_tree")

        if prune:
            G = self.prune_unconnected_parts(G, src_node, sink_node)
        self._check_graph(G)

        if return_subgraph_dict:
            subgraphs_dict = collections.OrderedDict(
                reversed(list(subgraphs_dict.items()))
            )
            if prune:
                for v in subgraphs_dict.values():
                    first_nodes = {e[0] for e in v.edges}
                    second_nodes = {e[1] for e in v.edges}
                    (vG_src_node,) = first_nodes - second_nodes
                    (vG_sink_node,) = second_nodes - first_nodes
                    v = self.prune_unconnected_parts(v, vG_src_node, vG_sink_node)
                    self._check_graph(v)
            return G, subgraphs_dict
        else:
            return G

    def prune_graph(self, graph: nx.DiGraph | Graph = None, edge_attr: bool = True):
        use_self = graph is None
        if use_self:
            graph = self

        in_degree = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        if len(in_degree) != 1:
            raise Exception(f"Multiple in degree nodes: {in_degree}")
        else:
            src_node = in_degree[0]
        out_degree = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        if len(out_degree) != 1:
            raise Exception(f"Multiple out degree nodes: {out_degree}")
        else:
            tgt_node = out_degree[0]

        if edge_attr:
            # remove edges with none
            remove_edge_list = []
            for u, v, edge_data in graph.edges.data():
                if isinstance(edge_data.op, Graph):
                    self.prune_graph(edge_data.op, edge_attr=edge_attr)
                elif isinstance(edge_data.op, list):
                    for op in edge_data.op:
                        if isinstance(op, Graph):
                            self.prune_graph(op, edge_attr=edge_attr)
                elif isinstance(edge_data.op, AbstractPrimitive) or issubclass(
                    edge_data.op, AbstractPrimitive
                ):
                    try:
                        if any(zero_op in edge_data.op_name for zero_op in self.zero_op):
                            remove_edge_list.append((u, v))
                    except TypeError:
                        if any(
                            zero_op in edge_data.op.get_op_name
                            for zero_op in self.zero_op
                        ):
                            remove_edge_list.append((u, v))
                elif inspect.isclass(edge_data.op):
                    assert not issubclass(
                        edge_data.op, Graph
                    ), "Found non-initialized graph. Abort."
                    # we look at an uncomiled op
                else:
                    raise ValueError(f"Unknown format of op: {edge_data.op}")
            # remove_edge_list = [
            #     e for e in graph.edges(data=True) if e[-1]["op_name"] in self.zero_op
            # ]
            graph.remove_edges_from(remove_edge_list)
        else:
            for n in list(nx.topological_sort(graph)):
                if n in graph.nodes():
                    if (
                        graph.nodes[n]["op_name"] in self.zero_op
                        or graph.nodes[n]["op_name"] in self.identity_op
                    ):

                        if graph.nodes[n]["op_name"] in self.identity_op:
                            # reconnect edges for removed nodes with 'skip_connect'
                            graph.add_edges_from(
                                [
                                    (e_i[0], e_o[1])
                                    for e_i in graph.in_edges(n)
                                    for e_o in graph.out_edges(n)
                                ]
                            )
                        # remove nodes with 'skip_connect' or 'none' label
                        graph.remove_node(n)

        graph = self.prune_unconnected_parts(graph, src_node, tgt_node)

        if not use_self:
            return graph

    @staticmethod
    def prune_unconnected_parts(graph, src_node, tgt_node):
        def _backtrack_remove(graph, node: int):
            predecessors = collections.deque(graph.predecessors(node))
            graph.remove_node(node)
            while len(predecessors) > 0:
                u = predecessors.pop()
                if u not in graph.nodes():  # if it is already removed skip
                    continue
                if (
                    len(list(graph.successors(u))) > 0
                ):  # there are more edges that could be valid paths
                    continue
                graph = _backtrack_remove(graph, u)
            return graph

        # after removal, some op nodes have no input nodes and some have no output nodes
        # --> remove these redundant nodes
        # O(|V|^2), but mostly O(|V|) (no zero op)
        for n in list(nx.topological_sort(graph)):
            if n in graph.nodes():
                predecessors = list(graph.predecessors(n))
                successors = list(graph.successors(n))
                if n != src_node and len(predecessors) == 0:
                    graph.remove_node(n)
                elif n != tgt_node and len(successors) == 0:
                    graph = _backtrack_remove(graph, n)
        return graph

    def _sampler_maxMin(self, largest: bool = True) -> str | list[str]:
        """Samples new parse tree(s) based on grammars.
        Assumes that the first rule of each production leads to
        smallest DAG and last to largest DAG!

        Args:
            largest (bool, optional): To find largest DAG, set to True. For smallest DAG set to False. Defaults to True.

        Returns:
            Union[str, List[str]]: Parse tree or list of parse trees
        """
        trees = [
            grammar.sampler_maxMin_func(grammar.start(), largest) + ")"
            for grammar in self.grammars
        ]
        return trees if len(trees) > 1 else trees[0]
