import collections
import inspect
import queue
from abc import abstractmethod
from copy import deepcopy
from functools import partial

import networkx as nx
import numpy as np

from .cfg import Grammar
from .graph import Graph
from .primitives import AbstractPrimitive
from .topologies import AbstractTopology


def get_edge_lists_of_topologies(terminal_map: dict) -> dict:
    topology_edge_lists = {}
    for k, v in terminal_map.items():
        if inspect.isclass(v):
            is_topology = issubclass(v, AbstractTopology)
        elif isinstance(v, partial):
            is_topology = issubclass(v.func, AbstractTopology)  # type: ignore[arg-type]
        else:
            is_topology = False
        if is_topology:
            if isinstance(v, partial):
                if hasattr(v.func, "get_edge_list"):
                    func_args = inspect.getfullargspec(v.func.get_edge_list).args  # type: ignore[attr-defined]
                    kwargs = {k: v for k, v in v.keywords.items() if k in func_args}
                    topology_edge_lists[k] = v.func.get_edge_list(**kwargs)  # type: ignore[attr-defined]
                elif hasattr(v.func, "edge_list"):
                    topology_edge_lists[k] = v.func.edge_list  # type: ignore[attr-defined]
                else:
                    raise Exception(
                        f"Please implement a get_edge_list static method for {v.func.__name__} or set edge_list!"
                    )
            else:
                topology_edge_lists[k] = v.edge_list
    return topology_edge_lists


class CoreGraphGrammar(Graph):
    def __init__(
        self,
        grammars: list[Grammar] | Grammar,
        terminal_to_op_names: dict,
        terminal_to_graph_edges: dict | None = None,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list | None = None,
        identity_op: list | None = None,
        name: str | None = None,
        scope: str | None = None,
        return_all_subgraphs: bool = False,
        return_graph_per_hierarchy: bool = False,
    ):
        super().__init__(name, scope)

        self.grammars = [grammars] if isinstance(grammars, Grammar) else grammars

        self.terminal_to_op_names = terminal_to_op_names

        grammar_terminals = {
            terminal for grammar in self.grammars for terminal in grammar.terminals
        }
        diff_terminals = grammar_terminals - set(self.terminal_to_op_names.keys())
        if len(diff_terminals) != 0:
            raise Exception(
                f"Terminals {diff_terminals} not defined in primitive mapping!"
            )

        if terminal_to_graph_edges is None:  # only compute it once -> more efficient
            self.terminal_to_graph_edges = get_edge_lists_of_topologies(
                self.terminal_to_op_names
            )
        else:
            self.terminal_to_graph_edges = terminal_to_graph_edges
        self.edge_attr = edge_attr
        self.edge_label = edge_label

        self.zero_op = zero_op if zero_op is not None else []
        self.identity_op = identity_op if identity_op is not None else []

        self.terminal_to_graph_nodes: dict = {}

        self.return_all_subgraphs = return_all_subgraphs
        self.return_graph_per_hierarchy = return_graph_per_hierarchy

    def clear_graph(self):
        while len(self.nodes()) != 0:
            self.remove_node(next(iter(self.nodes())))

    @abstractmethod
    def id_to_string_tree(self, identifier: str):
        raise NotImplementedError

    @abstractmethod
    def string_tree_to_id(self, string_tree: str):
        raise NotImplementedError

    @abstractmethod
    def compute_prior(self, log: bool = True):
        raise NotImplementedError

    @abstractmethod
    def compose_functions(self, flatten_graph: bool = True):
        raise NotImplementedError

    @staticmethod
    def _check_graph(graph: nx.DiGraph):
        if len(graph) == 0 or graph.number_of_edges() == 0:
            raise ValueError("Invalid DAG")

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
        valid_terminals: collections.abc.KeysView,
        edge_attr: bool = True,
        sym_name: str = "op_name",
        prune: bool = True,
        add_subtree_map: bool = False,
        return_all_subgraphs: bool | None = None,
        return_graph_per_hierarchy: bool | None = None,
    ) -> nx.DiGraph | tuple[nx.DiGraph, collections.OrderedDict]:
        """Generates graph from parse tree in string representation.
        Note that we ignore primitive HPs!

        Args:
            string_tree (str): parse tree.
            grammar (Grammar): underlying grammar.
            valid_terminals (list): list of keys.
            edge_attr (bool, optional): Shoud graph be edge attributed (True) or node attributed (False). Defaults to True.
            sym_name (str, optional): Attribute name of operation. Defaults to "op_name".
            prune (bool, optional): Prune graph, e.g., None operations etc. Defaults to True.
            add_subtree_map (bool, optional): Add attribute indicating to which subtrees of
                the parse tree the specific part belongs to. Can only be true if you set prune=False!
                TODO: Check if we really need this constraint or can also allow pruning. Defaults to False.
            return_all_subgraphs (bool, optional): Additionally returns an hierarchical dictionary
                containing all subgraphs. Defaults to False.
                TODO: check if edge attr also works.
            return_graph_per_hierarchy (bool, optional): Additionally returns a graph from each
                each hierarchy.

        Returns:
            nx.DiGraph: [description]
        """

        def get_node_labels(graph: nx.DiGraph):
            return [
                (n, d[sym_name])
                for n, d in graph.nodes(data=True)
                if d[sym_name] != "input" and d[sym_name] != "output"
            ]

        def get_hierarchicy_dict(
            string_tree: str,
            subgraphs: dict,
            hierarchy_dict: dict | None = None,
            hierarchy_level_counter: int = 0,
        ):
            if hierarchy_dict is None:
                hierarchy_dict = {}
            if hierarchy_level_counter not in hierarchy_dict:
                hierarchy_dict[hierarchy_level_counter] = []
            hierarchy_dict[hierarchy_level_counter].append(string_tree)
            node_labels = get_node_labels(subgraphs[string_tree])
            for _, node_label in node_labels:
                if node_label in subgraphs:
                    hierarchy_dict = get_hierarchicy_dict(
                        node_label, subgraphs, hierarchy_dict, hierarchy_level_counter + 1
                    )
            return hierarchy_dict

        def get_graph_per_hierarchy(string_tree: str, subgraphs: dict):
            hierarchy_dict = get_hierarchicy_dict(
                string_tree=string_tree, subgraphs=subgraphs
            )

            graph_per_hierarchy = collections.OrderedDict()
            for k, v in hierarchy_dict.items():
                if k == 0:
                    graph_per_hierarchy[k] = subgraphs[v[0]]
                else:
                    subgraph_ = graph_per_hierarchy[k - 1].copy()
                    node_labels = get_node_labels(subgraph_)
                    for node, node_label in node_labels:
                        if node_label in list(subgraphs.keys()):
                            in_nodes = list(subgraph_.predecessors(node))
                            out_nodes = list(subgraph_.successors(node))
                            node_offset = max(subgraph_.nodes) + 1

                            new_subgraph = nx.relabel.relabel_nodes(
                                subgraphs[node_label],
                                mapping={
                                    n: n + node_offset
                                    for n in subgraphs[node_label].nodes
                                },
                                copy=True,
                            )
                            first_nodes = {e[0] for e in new_subgraph.edges}
                            second_nodes = {e[1] for e in new_subgraph.edges}
                            (begin_node,) = first_nodes - second_nodes
                            (end_node,) = second_nodes - first_nodes
                            successors = list(new_subgraph.successors(begin_node))
                            predecessors = list(new_subgraph.predecessors(end_node))
                            new_subgraph.remove_nodes_from([begin_node, end_node])
                            edges = []
                            added_identities = False
                            for in_node in in_nodes:
                                for succ in successors:
                                    if succ == end_node:
                                        if not added_identities:
                                            edges.extend(
                                                [
                                                    (inn, onn)
                                                    for inn in in_nodes
                                                    for onn in out_nodes
                                                ]
                                            )
                                        added_identities = True
                                    else:
                                        edges.append((in_node, succ))
                            for out_node in out_nodes:
                                for pred in predecessors:
                                    if pred != begin_node:
                                        edges.append((pred, out_node))

                            subgraph_ = nx.compose(new_subgraph, subgraph_)
                            subgraph_.add_edges_from(edges)
                            subgraph_.remove_node(node)

                    graph_per_hierarchy[k] = subgraph_
            return graph_per_hierarchy

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

                for e_ in filter(lambda e: (e[1] == u), edge_list):
                    node_list.append((edge_to_node_map[e_], ni))

            return node_list, edge_to_node_map

        def skip_char(char: str) -> bool:
            return char in [" ", "\t", "\n", "[", "]"]

        if prune:
            add_subtree_map = False

        if return_all_subgraphs is None:
            return_all_subgraphs = self.return_all_subgraphs
        if return_graph_per_hierarchy is None:
            return_graph_per_hierarchy = self.return_graph_per_hierarchy
        compute_subgraphs = return_all_subgraphs or return_graph_per_hierarchy

        G = nx.DiGraph()
        if add_subtree_map:
            q_nonterminals: collections.deque = collections.deque()
        if compute_subgraphs:
            q_subtrees: collections.deque = collections.deque()
            q_subgraphs: collections.deque = collections.deque()
            subgraphs_dict = collections.OrderedDict()
        if edge_attr:
            node_offset = 0
            q_el: collections.deque = collections.deque()  # edge-attr
            terminal_to_graph = self.terminal_to_graph_edges
        else:  # node-attributed
            G.add_node(0, **{sym_name: "input"})
            G.add_node(1, **{sym_name: "output"})
            node_offset = 2
            if bool(self.terminal_to_graph_nodes):
                terminal_to_graph_nodes = self.terminal_to_graph_nodes
            else:
                terminal_to_graph_nodes = {
                    k: to_node_attributed_edge_list(edge_list) if edge_list else []
                    for k, edge_list in self.terminal_to_graph_edges.items()
                }
                self.terminal_to_graph_nodes = terminal_to_graph_nodes
            terminal_to_graph = {
                k: v[0] if v else [] for k, v in terminal_to_graph_nodes.items()
            }
            q_el = collections.deque()  # node-attr

        # pre-compute stuff
        begin_end_nodes = {}
        for sym, g in terminal_to_graph.items():
            if g:
                first_nodes = {e[0] for e in g}
                second_nodes = {e[1] for e in g}
                (begin_node,) = first_nodes - second_nodes
                (end_node,) = second_nodes - first_nodes
                begin_end_nodes[sym] = (begin_node, end_node)
            else:
                begin_end_nodes[sym] = (None, None)

        for split_idx, sym in enumerate(string_tree.split(" ")):
            is_nonterminal = False
            if sym == "":
                continue
            if compute_subgraphs:
                new_sym = True
                sym_copy = sym[:]
            if sym[0] == "(":
                sym = sym[1:]
                is_nonterminal = True
            if sym[-1] == ")":
                if add_subtree_map:
                    for _ in range(sym.count(")")):
                        q_nonterminals.pop()
                if compute_subgraphs:
                    new_sym = False
                while sym[-1] == ")" and sym not in valid_terminals:
                    sym = sym[:-1]

            if compute_subgraphs and new_sym:
                if sym in grammar.nonterminals:
                    # need dict as a graph can have multiple subgraphs
                    q_subtrees.append(sym_copy[:])
                else:
                    q_subtrees[-1] += f" {sym_copy}"

            if len(sym) == 1 and skip_char(sym[0]):
                continue

            if add_subtree_map and sym in grammar.nonterminals:
                q_nonterminals.append((sym, split_idx))
            elif sym in valid_terminals and not is_nonterminal:  # terminal symbol
                if sym in self.terminal_to_graph_edges:
                    if len(q_el) == 0:
                        if edge_attr:
                            edges = [
                                tuple(t + node_offset for t in e)
                                for e in self.terminal_to_graph_edges[sym]
                            ]
                        else:  # node-attr
                            edges = [
                                tuple(t for t in e)
                                for e in terminal_to_graph_nodes[sym][0]
                            ]
                            nodes = [
                                terminal_to_graph_nodes[sym][1][e]
                                for e in self.terminal_to_graph_edges[sym]
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
                                for e in self.terminal_to_graph_edges[sym]
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
                                for e in self.terminal_to_graph_edges[sym]
                            ]

                    G.add_edges_from(edges)

                    if compute_subgraphs:
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
                        node_offset += max(max(self.terminal_to_graph_edges[sym]))
                    else:
                        node_offset += max(terminal_to_graph_nodes[sym][1].values())
                else:  # primitive operations
                    el = q_el.pop()
                    if edge_attr:
                        u, v = el
                        if prune and sym in self.zero_op:
                            G.remove_edge(u, v)
                            if compute_subgraphs:
                                q_subgraphs[-1]["graph"].remove_edge(u, v)
                                del q_subgraphs[-1]["atoms"][(u, v)]
                        else:
                            G[u][v][sym_name] = sym
                            if compute_subgraphs:
                                q_subgraphs[-1]["graph"][u][v][sym_name] = sym
                            if add_subtree_map:
                                G[u][v]["subtrees"].append(q_nonterminals[-1])
                                q_nonterminals.pop()
                    else:  # node-attr
                        n = el
                        if prune and sym in self.zero_op:
                            G.remove_node(n)
                            if compute_subgraphs:
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
                            if compute_subgraphs:
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
                            if compute_subgraphs:
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
            if compute_subgraphs and sym_copy[-1] == ")":
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

        if return_all_subgraphs or return_graph_per_hierarchy:
            return_val = [G]
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
            if return_all_subgraphs:
                return_val.append(subgraphs_dict)
            if return_graph_per_hierarchy:
                graph_per_hierarchy = get_graph_per_hierarchy(string_tree, subgraphs_dict)
                _ = (
                    graph_per_hierarchy.popitem()
                )  # remove last graph since it is equal to full graph
                return_val.append(graph_per_hierarchy)
            return return_val
        return G

    def get_graph_representation(
        self,
        identifier: str,
        grammar: Grammar,
        edge_attr: bool,
    ) -> nx.DiGraph:
        """This functions takes an identifier and constructs the
        (multi-variate) composition of the functions it describes.

        Args:
            identifier (str): identifier
            grammar (Grammar): grammar
            flatten_graph (bool, optional): Whether to flatten the graph. Defaults to True.

        Returns:
            nx.DiGraph: (multi-variate) composition of functions.
        """

        def _skip_char(char: str) -> bool:
            return char in [" ", "\t", "\n", "[", "]"]

        def _get_sym_from_split(split: str) -> str:
            start_idx, end_idx = 0, len(split)
            while start_idx < end_idx and split[start_idx] == "(":
                start_idx += 1
            while start_idx < end_idx and split[end_idx - 1] == ")":
                end_idx -= 1
            return split[start_idx:end_idx]

        def to_node_attributed_edge_list(
            edge_list: list[tuple],
        ) -> tuple[list[tuple[int, int]], dict]:
            first_nodes = {e[0] for e in edge_list}
            second_nodes = {e[1] for e in edge_list}
            src = first_nodes - second_nodes
            tgt = second_nodes - first_nodes
            node_offset = len(src)
            edge_to_node_map = {e: i + node_offset for i, e in enumerate(edge_list)}
            node_list = []
            for e in edge_list:
                ni = edge_to_node_map[e]
                u, v = e
                if u in src:
                    node_list.append((u, ni))
                if v in tgt:
                    node_list.append((ni, v))

                for e_ in filter(lambda e: (e[1] == u), edge_list):
                    node_list.append((edge_to_node_map[e_], ni))

            return node_list, edge_to_node_map

        descriptor = self.id_to_string_tree(identifier)

        if edge_attr:
            terminal_to_graph = self.terminal_to_graph_edges
        else:  # node-attr
            terminal_to_graph_nodes = {
                k: to_node_attributed_edge_list(edge_list) if edge_list else (None, None)
                for k, edge_list in self.terminal_to_graph_edges.items()
            }
            terminal_to_graph = {k: v[0] for k, v in terminal_to_graph_nodes.items()}
            # edge_to_node_map = {k: v[1] for k, v in terminal_to_graph_nodes.items()}

        q_nonterminals: queue.LifoQueue = queue.LifoQueue()
        q_topologies: queue.LifoQueue = queue.LifoQueue()
        q_primitives: queue.LifoQueue = queue.LifoQueue()

        G = nx.DiGraph()
        for _, split in enumerate(descriptor.split(" ")):
            if _skip_char(split):
                continue
            sym = _get_sym_from_split(split)

            if sym in grammar.terminals:
                is_topology = False
                if (
                    inspect.isclass(self.terminal_to_op_names[sym])
                    and issubclass(self.terminal_to_op_names[sym], AbstractTopology)
                    or isinstance(self.terminal_to_op_names[sym], partial)
                    and issubclass(self.terminal_to_op_names[sym].func, AbstractTopology)
                ):
                    is_topology = True

                if is_topology:
                    q_topologies.put([self.terminal_to_op_names[sym], 0])
                else:  # is primitive operation
                    q_primitives.put(self.terminal_to_op_names[sym])
                    q_topologies.queue[-1][1] += 1  # count number of primitives
            elif sym in grammar.nonterminals:
                q_nonterminals.put(sym)
            else:
                raise Exception(f"Unknown symbol {sym}")

            if ")" in split:
                # closing symbol of production
                while ")" in split:
                    if q_nonterminals.qsize() == q_topologies.qsize():
                        topology, number_of_primitives = q_topologies.get(block=False)
                        primitives = [
                            q_primitives.get(block=False)
                            for _ in range(number_of_primitives)
                        ][::-1]
                        if (
                            topology in terminal_to_graph
                            and terminal_to_graph[topology] is not None
                        ) or isinstance(topology, partial):
                            raise NotImplementedError
                        else:
                            composed_function = topology(*primitives)
                            node_attr_dag = composed_function.get_node_list_and_ops()
                            G = node_attr_dag  # TODO only works for DARTS for now

                        if not q_topologies.empty():
                            q_primitives.put(composed_function)
                            q_topologies.queue[-1][1] += 1

                    _ = q_nonterminals.get(block=False)
                    split = split[:-1]

        if not q_topologies.empty():
            raise Exception("Invalid descriptor")

        # G = self.prune_unconnected_parts(G, src_node, sink_node)
        # self._check_graph(G)
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
                if n in graph.nodes() and (
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
        return None

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

    @staticmethod
    def flatten_graph(
        graph: nx.DiGraph,
        flattened_graph: Graph = None,
        start_node: int | None = None,
        end_node: int | None = None,
    ):
        if flattened_graph is None:
            flattened_graph = Graph()
        nodes: dict = {}
        for u, v, data in graph.edges(data=True):
            if u in nodes:
                _u = nodes[u]
            else:
                _u = (
                    1
                    if len(flattened_graph.nodes.keys()) == 0  # type: ignore[union-attr]
                    else max(flattened_graph.nodes.keys()) + 1  # type: ignore[union-attr]
                )
                _u = (
                    start_node
                    if graph.in_degree(u) == 0 and start_node is not None
                    else _u
                )
                nodes[u] = _u
                if _u not in flattened_graph.nodes:  # type: ignore[union-attr]
                    flattened_graph.add_node(_u)  # type: ignore[union-attr]
                    flattened_graph.nodes[_u].update(graph.nodes[u])  # type: ignore[union-attr]

            if v in nodes:
                _v = nodes[v]
            else:
                _v = max(flattened_graph.nodes.keys()) + 1  # type: ignore[union-attr]
                _v = end_node if graph.out_degree(v) == 0 and end_node is not None else _v
                nodes[v] = _v
                if _v not in flattened_graph.nodes:  # type: ignore[union-attr]
                    flattened_graph.add_node(_v)  # type: ignore[union-attr]
                flattened_graph.nodes[_v].update(  # type: ignore[union-attr]
                    graph.nodes[v]
                )  # last time node is called combo op is used

            if isinstance(data["op"], Graph):
                flattened_graph = CoreGraphGrammar.flatten_graph(
                    data["op"], flattened_graph, start_node=_u, end_node=_v
                )
            else:
                flattened_graph.add_edge(_u, _v)  # type: ignore[union-attr]
                flattened_graph.edges[_u, _v].update(data)  # type: ignore[union-attr]

        return flattened_graph

    def _compose_functions(
        self, identifier: str, grammar: Grammar, flatten_graph: bool = True
    ) -> nx.DiGraph:
        """This functions takes an identifier and constructs the
        (multi-variate) composition of the functions it describes.

        Args:
            identifier (str): identifier
            grammar (Grammar): grammar
            flatten_graph (bool, optional): Whether to flatten the graph. Defaults to True.

        Returns:
            nx.DiGraph: (multi-variate) composition of functions
        """
        descriptor = self.id_to_string_tree(identifier)

        symbols = grammar.nonterminals + grammar.terminals
        max_match = max(map(len, symbols))
        find_longest_match_func = partial(
            find_longest_match,
            descriptor=descriptor,
            symbols=symbols,
            max_match=max_match,
        )

        q_nonterminals: queue.LifoQueue = queue.LifoQueue()
        q_topologies: queue.LifoQueue = queue.LifoQueue()
        q_primitives: queue.LifoQueue = queue.LifoQueue()
        i = 0
        while i < len(descriptor):
            char = descriptor[i]
            if skip_char(char, descriptor, i):
                pass
            elif char == ")" and descriptor[i - 1] != " ":
                # closing symbol of production
                if q_nonterminals.qsize() == q_topologies.qsize():
                    topology, number_of_primitives = q_topologies.get(block=False)
                    primitives = [
                        q_primitives.get(block=False) for _ in range(number_of_primitives)
                    ][::-1]
                    composed_function = topology(*primitives)
                    if not q_topologies.empty():
                        q_primitives.put(composed_function)
                        q_topologies.queue[-1][1] += 1
                _ = q_nonterminals.get(block=False)
            else:
                j = find_longest_match_func(i)
                sym = descriptor[i:j]
                i = j - 1

                if sym in grammar.terminals and descriptor[i - 1] != "(":
                    is_topology = False
                    if (
                        inspect.isclass(self.terminal_to_op_names[sym])
                        and issubclass(self.terminal_to_op_names[sym], AbstractTopology)
                        or isinstance(self.terminal_to_op_names[sym], partial)
                        and issubclass(
                            self.terminal_to_op_names[sym].func, AbstractTopology
                        )
                    ):
                        is_topology = True

                    if is_topology:
                        q_topologies.put([self.terminal_to_op_names[sym], 0])
                    else:  # is primitive operation
                        q_primitives.put(self.terminal_to_op_names[sym])
                        q_topologies.queue[-1][1] += 1  # count number of primitives
                elif sym in grammar.nonterminals:
                    q_nonterminals.put(sym)
                else:
                    raise Exception(f"Unknown symbol {sym}")

            i += 1

        if not q_topologies.empty():
            raise Exception("Invalid descriptor")

        self._check_graph(composed_function)

        if flatten_graph:
            composed_function = self.flatten_graph(composed_function)

        return composed_function

    def graph_to_self(self, graph: nx.DiGraph, clear_self: bool = True) -> None:
        """Copies graph to self.

        Args:
            graph (nx.DiGraph): graph
        """
        if clear_self:
            self.clear()
        for u, v, data in graph.edges(data=True):
            self.add_edge(u, v)  # type: ignore[union-attr]
            self.edges[u, v].update(data)  # type: ignore[union-attr]
        for n, data in graph.nodes(data=True):
            self.nodes[n].update(**data)


def skip_char(char: str, descriptor: str, i: int) -> bool:
    if char in [" ", "\t", "\n"]:
        return True
    # special case: "(" is (part of) a terminal
    if i != 0 and char == "(" and descriptor[i - 1] == " " and descriptor[i + 1] == " ":
        return False
    return char == "("


def find_longest_match(
    i: int, descriptor: str, symbols: list[str], max_match: int
) -> int:
    # search for longest matching symbol and add it
    # assumes that the longest match is the true match
    j = min(i + max_match, len(descriptor) - 1)
    while j > i and j < len(descriptor):
        if descriptor[i:j] in symbols:
            break
        j -= 1
    if j == i:
        raise Exception(f"Terminal or nonterminal at position {i} does not exist")
    return j
