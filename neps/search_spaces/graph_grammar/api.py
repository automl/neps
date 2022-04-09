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


class FunctionParameter(GraphGrammar):
    def __init__(
        self,
        build_fn: Callable,
        structure: Grammar | str | dict,
        primitives,
        name: str = "",
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

        self.build: Callable = build_fn
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

        self.build(self)

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
            build_fn=self.build,
            structure=self.grammars[0],
            primitives=self.terminal_to_op_names,
            name=self.name,
        )
        g.string_tree = child
        g.id = child
        return g
