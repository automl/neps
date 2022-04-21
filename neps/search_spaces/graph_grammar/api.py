from __future__ import annotations

import inspect
from typing import Callable

import networkx as nx

from .cfg import Grammar
from .cfg_variants.constrained_cfg import ConstrainedGrammar
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
        structure: Grammar | ConstrainedGrammar | str | dict,
        primitives: dict,
        constraint_kwargs: dict | None = None,
        name: str = "",
        set_recursive_attribute: Callable | None = None,
        old_build_api: bool = False,
        **kwargs,
    ):
        local_vars = locals()
        self.input_kwargs = {
            args: local_vars[args]
            for args in inspect.getfullargspec(self.__init__).args  # type: ignore[misc]
            if args != "self"
        }
        self.input_kwargs.update(**kwargs)

        if isinstance(structure, dict):
            structure = _dict_structure_to_str(structure, primitives)
        if isinstance(structure, str):
            if constraint_kwargs is None:
                structure = Grammar.fromstring(structure)
            else:
                structure = ConstrainedGrammar.fromstring(structure)
                structure.set_constraints(**constraint_kwargs)

        super().__init__(
            grammar=structure,  # type: ignore[arg-type]
            terminal_to_op_names=primitives,
            edge_attr=False,
            **kwargs,
        )

        self._set_recursive_attribute = set_recursive_attribute
        self._old_build_api = old_build_api
        self.name: str = name

    def setup(self):
        composed_function = self.compose_functions(self.id)
        self.graph_to_self(composed_function)
        self.prune_graph()

        if self._old_build_api:
            self._set_recursive_attribute(self)  # type: ignore[misc] # This is the full build_fn
        elif self._set_recursive_attribute:
            _build(self, self._set_recursive_attribute)

        self.compile()
        self.update_op_names()

    def to_pytorch(self):
        self.clear_graph()
        if len(self.nodes()) == 0:
            self.setup()
        return super().to_pytorch()

    def to_tensorflow(self, inputs):
        composed_function = self.compose_functions(self.id, flatten_graph=False)
        return composed_function(inputs)

    def create_new_instance_from_id(self, identifier: str):
        g = FunctionParameter(**self.input_kwargs)  # type: ignore[arg-type]
        g.load_from(identifier)
        return g
