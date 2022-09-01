from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Callable

import networkx as nx
from torch import nn

from .cfg import Grammar
from .cfg_variants.constrained_cfg import ConstrainedGrammar
from .graph import Graph
from .graph_grammar import GraphGrammar, GraphGrammarMultipleRepetitive


def _dict_structure_to_str(structure: dict, primitives: dict) -> str:
    grammar = ""
    for nonterminal, productions in structure.items():
        grammar += nonterminal + " -> " + " | ".join(productions) + "\n"
    for primitive in primitives.keys():
        grammar = grammar.replace(f" {primitive} ", f' "{primitive}" ')
        grammar = grammar.replace(f" {primitive}\n", f' "{primitive}"\n')
    return grammar


def _build(graph, set_recursive_attribute) -> Graph:
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
    return graph


def FunctionParameter(**kwargs):
    """Factory function"""

    if "structure" not in kwargs:
        raise ValueError("Factory function requires structure")
    if not isinstance(kwargs["structure"], list) or len(kwargs["structure"]) == 1:
        base = GraphGrammar
    else:
        base = GraphGrammarMultipleRepetitive

    class _FunctionParameter(base):
        def __init__(
            self,
            structure: Grammar
            | list[Grammar]
            | ConstrainedGrammar
            | list[ConstrainedGrammar]
            | str
            | list[str]
            | dict
            | list[dict],
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

            if isinstance(structure, list):
                structures = [
                    _dict_structure_to_str(st, primitives) if isinstance(st, dict) else st
                    for st in structure
                ]
                _structures = []
                for st in structures:
                    if isinstance(st, str):
                        if constraint_kwargs is None:
                            _st = Grammar.fromstring(st)
                        else:
                            _st = ConstrainedGrammar.fromstring(st)
                            _st.set_constraints(**constraint_kwargs)
                    _structures.append(_st)  # type: ignore[has-type]
                structures = _structures

                super().__init__(
                    grammars=structures,
                    terminal_to_op_names=primitives,
                    edge_attr=False,
                    **kwargs,
                )
            else:
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

        def to_pytorch(self) -> nn.Module:
            self.clear_graph()
            if len(self.nodes()) == 0:
                composed_function = self.compose_functions()
                # part below is required since PyTorch has no standard functional API
                self.graph_to_self(composed_function)
                self.prune_graph()

                if self._old_build_api:
                    arch = self._set_recursive_attribute(deepcopy(self))  # type: ignore[misc] # This is the full build_fn
                elif self._set_recursive_attribute:
                    arch = _build(deepcopy(self), self._set_recursive_attribute)

                arch.compile()
                arch.update_op_names()
            return arch.to_pytorch()  # create PyTorch model

        def to_tensorflow(self, inputs):
            composed_function = self.compose_functions(flatten_graph=False)
            return composed_function(inputs)

        def create_new_instance_from_id(self, identifier: str):
            g = FunctionParameter(**self.input_kwargs)  # type: ignore[arg-type]
            g.load_from(identifier)
            return g

        def copy(self):
            return deepcopy(self)

    return _FunctionParameter(**kwargs)
