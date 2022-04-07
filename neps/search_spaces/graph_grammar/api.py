from __future__ import annotations

from typing import Callable

import networkx as nx

from .cfg import Grammar
from .graph_grammar import GraphGrammar


class FunctionParameter(GraphGrammar):
    def __init__(
        self,
        build_fn: Callable,
        grammar: Grammar | str,
        terminal_to_op_names,
        name: str = "",
    ):
        if isinstance(grammar, str):
            grammar = Grammar.fromstring(grammar)

        super().__init__(
            grammar=grammar,
            terminal_to_op_names=terminal_to_op_names,
            edge_attr=False,
        )

        self.build = build_fn
        self.name: str = name
        self.nxTree = None
        self.string_tree = ""
        self.id = ""

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
        pytorch_model = super().to_pytorch()
        return pytorch_model

    def create_graph_from_string(self, child: str):
        g = FunctionParameter(
            grammar=self.grammars[0],
            build_fn=self.build_fn,
            terminal_to_op_names=self.terminal_to_op_names,
            name=self.name,
        )
        g.string_tree = child
        g.id = child
        g.create_representation(g.string_tree)
        return g
