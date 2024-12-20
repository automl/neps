from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

import networkx as nx

from .cfg import Grammar
from .cfg_variants.constrained_cfg import ConstrainedGrammar
from .graph_grammar import GraphGrammar

if TYPE_CHECKING:
    from torch import nn


def _dict_structure_to_str(
    structure: dict, primitives: dict, repetitive_mapping: dict | None = None
) -> str:
    def _save_replace(string: str, __old: str, __new: str):
        while string.count(__old) > 0:
            string = string.replace(__old, __new)
        return string

    grammar = ""
    for nonterminal, productions in structure.items():
        grammar += nonterminal + " -> " + " | ".join(productions) + "\n"
    grammar = grammar.replace("(", " ")
    grammar = grammar.replace(")", "")
    grammar = grammar.replace(",", "")
    for primitive in primitives:
        grammar = _save_replace(grammar, f" {primitive} ", f' "{primitive}" ')
        grammar = _save_replace(grammar, f" {primitive}\n", f' "{primitive}"\n')
    if repetitive_mapping is not None:
        for placeholder in repetitive_mapping:
            grammar = _save_replace(grammar, f" {placeholder} ", f' "{placeholder}" ')
            grammar = _save_replace(grammar, f" {placeholder}\n", f' "{placeholder}"\n')
    return grammar


def _build(graph, set_recursive_attribute):
    in_node = next(n for n in graph.nodes if graph.in_degree(n) == 0)
    for n in nx.topological_sort(graph):
        for pred in graph.predecessors(n):
            e = (pred, n)
            op_name = graph.edges[e]["op_name"]
            if pred == in_node:
                predecessor_values = None
            else:
                pred_pred = next(iter(graph.predecessors(pred)))
                predecessor_values = graph.edges[(pred_pred, pred)]
            graph.edges[e].update(set_recursive_attribute(op_name, predecessor_values))


def Architecture(**kwargs):
    """Factory function."""
    if "structure" not in kwargs:
        raise ValueError("Factory function requires structure")
    if not isinstance(kwargs["structure"], list) or len(kwargs["structure"]) == 1:
        base = GraphGrammar

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
            # TODO: Follow this rabbit hole for `constraint_kwargs`,
            # it can all be deleted my friend
            constraint_kwargs: dict | None = None,
            name: str = "ArchitectureParameter",
            set_recursive_attribute: Callable | None = None,
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
                    _dict_structure_to_str(
                        st,
                        primitives,
                        repetitive_mapping=kwargs.get(
                            "terminal_to_sublanguage_map", None
                        ),
                    )
                    if isinstance(st, dict)
                    else st
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
                        structure.set_constraints(**constraint_kwargs)  # type: ignore[union-attr]

                super().__init__(
                    grammar=structure,  # type: ignore[arg-type]
                    terminal_to_op_names=primitives,
                    edge_attr=False,
                    **kwargs,
                )

            self._set_recursive_attribute = set_recursive_attribute
            self.name: str = name

        def to_pytorch(self) -> nn.Module:
            self.clear_graph()
            if len(self.nodes()) == 0:
                composed_function = self.compose_functions()
                # part below is required since PyTorch has no standard functional API
                self.graph_to_self(composed_function)
                self.prune_graph()

                if self._set_recursive_attribute:
                    m = _build(self, self._set_recursive_attribute)

                if m is not None:
                    return m

                self.compile()
                self.update_op_names()
            return super().to_pytorch()  # create PyTorch model

        def create_new_instance_from_id(self, identifier: str):
            g = Architecture(**self.input_kwargs)  # type: ignore[arg-type]
            g.load_from(identifier)
            return g

    return _FunctionParameter(**kwargs)


def ArchitectureParameter(**kwargs):
    """Deprecated: Use `Architecture` instead of `ArchitectureParameter`.

    This function remains for backward compatibility and will raise a deprecation
    warning if used.
    """
    import warnings

    warnings.warn(
        (
            "Usage of 'neps.ArchitectureParameter' is deprecated and will be removed in"
            " future releases. Please use 'neps.Architecture' instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )

    return Architecture(**kwargs)


Function = Architecture


def FunctionParameter(**kwargs):
    """Deprecated: Use `Function` instead of `FunctionParameter`.

    This function remains for backward compatibility and will raise a deprecation
    warning if used.
    """
    import warnings

    warnings.warn(
        (
            "Usage of 'neps.FunctionParameter' is deprecated and will be removed in"
            " future releases. Please use 'neps.Function' instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )

    return Function(**kwargs)
