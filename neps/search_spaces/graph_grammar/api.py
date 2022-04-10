from __future__ import annotations

from typing import Callable

import networkx as nx

from .cfg import Grammar
from .graph_grammar import GraphGrammar


def _dict_structure_to_str(structure, primitives):
    grammar = ""
    for nonterminal, productions in structure.items():
        grammar += nonterminal + " -> " + " | ".join(productions) + "\n"
    for primitive in primitives.keys():
        grammar = grammar.replace(f" {primitive} ", f' "{primitive}" ')
        grammar = grammar.replace(f" {primitive}\n", f' "{primitive}"\n')
    return grammar


def _build(graph, set_recursive_attribute):
    in_node = [n for n in graph.nodes if graph.in_degree(n) == 0][0]
    for n in nx.topological_sort(graph):
        for pred in graph.predecessors(n):
            e = (pred, n)
            op_name = graph.edges[e]["op_name"]
            if pred == in_node:
                predecessor_values = None
            else:
                pred_pred = list(graph.predecessors(pred))[0]
                predecessor_values = graph.edges[(pred_pred, pred)]
            graph.edges[e].update(set_recursive_attribute(op_name, predecessor_values))


class FunctionParameter(GraphGrammar):
    def __init__(
        self,
        structure: Grammar | str | dict,
        primitives,
        name: str = "",
        set_recursive_attribute: Callable | None = None,
        old_build_api=False,
        **kwargs,
    ):
        if isinstance(structure, dict):
            structure = _dict_structure_to_str(structure, primitives)
        if isinstance(structure, str):
            structure = Grammar.fromstring(structure)

        super().__init__(
            grammar=structure,
            terminal_to_op_names=primitives,
            edge_attr=False,
            **kwargs,
        )

        self._set_recursive_attribute = set_recursive_attribute
        self._old_build_api = old_build_api
        self.name: str = name
        self.nxTree: nx.DiGraph = None
        self.string_tree: str = ""
        self.id: str = ""

    def setup(self, tree: nx.DiGraph):
        self.build_graph_from_tree(
            tree=tree,
            terminal_to_torch_map=self.terminal_to_op_names,
        )
        self.prune_graph()

        if self._old_build_api:
            self._set_recursive_attribute(self)  # This is the full build_fn
        elif self._set_recursive_attribute:
            _build(self, self._set_recursive_attribute)

        self.compile()
        self.update_op_names()

    def to_pytorch(self):
        self.clear_graph()
        if self.nxTree is None:
            self.nxTree = self.create_nx_tree(self.string_tree)
        if len(self.nodes()) == 0:
            self.setup(self.nxTree)
        return super().to_pytorch()

    def create_graph_from_string(self, child: str):
        g = FunctionParameter(
            set_recursive_attribute=self.build,
            structure=self.grammars[0],
            primitives=self.terminal_to_op_names,
            name=self.name,
        )
        g.string_tree = child
        g.id = child
        return g
