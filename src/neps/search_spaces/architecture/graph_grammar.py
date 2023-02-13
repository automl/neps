from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import networkx as nx
import numpy as np
from nltk import Nonterminal

from ..parameter import Parameter
from .cfg import Grammar
from .cfg_variants.constrained_cfg import ConstrainedGrammar
from .core_graph_grammar import CoreGraphGrammar
from .crossover import repetitive_search_space_crossover, simple_crossover
from .mutations import bananas_mutate, repetitive_search_space_mutation, simple_mutate


class GraphGrammar(CoreGraphGrammar, Parameter):
    hp_name = "graph_grammar"

    def __init__(  # pylint: disable=W0102
        self,
        grammar: Grammar,
        terminal_to_op_names: dict,
        prior: dict = None,
        terminal_to_graph_edges: dict = None,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        new_graph_repr_func: bool = False,
        name: str = None,
        scope: str = None,
        **kwargs,
    ):
        if isinstance(grammar, list) and len(grammar) != 1:
            raise NotImplementedError("Does not support multiple grammars")

        CoreGraphGrammar.__init__(
            self,
            grammars=grammar,
            terminal_to_op_names=terminal_to_op_names,
            terminal_to_graph_edges=terminal_to_graph_edges,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
            **kwargs,
        )
        Parameter.__init__(self, set_default_value=False)

        self.string_tree: str = ""
        self._function_id: str = ""
        self.nxTree: nx.DiGraph = None
        self._value: nx.DiGraph = None
        self.new_graph_repr_func = new_graph_repr_func

        if prior is not None:
            self.grammars[0].prior = prior
        self.has_prior = prior is not None

    @property
    def id(self):
        if self._function_id is None or self._function_id == "":
            if self.string_tree == "":
                raise ValueError("Cannot infer identifier!")
            self._function_id = self.string_tree_to_id(self.string_tree)
        return self._function_id

    @id.setter
    def id(self, value):
        self._function_id = value

    @staticmethod
    def id_to_string_tree(identifier: str):
        return identifier

    @staticmethod
    def string_tree_to_id(string_tree: str):
        return string_tree

    def __eq__(self, other):
        return self.id == other.id

    @property
    def search_space_size(self) -> int:
        return self.grammars[0].compute_space_size

    @abstractmethod
    def create_new_instance_from_id(self, identifier: str):
        raise NotImplementedError

    def reset(self):
        self.clear_graph()
        self.string_tree = ""
        self.nxTree = None
        self._value = None
        self._function_id = ""

    def compose_functions(self, flatten_graph: bool = True):
        return self._compose_functions(self.id, self.grammars[0], flatten_graph)

    def unparse_tree(self, identifier: str, as_composition: bool = True):
        return self._unparse_tree(identifier, self.grammars[0], as_composition)

    def get_dictionary(self) -> dict:
        return {"graph_grammar": self.id}

    def get_graphs(self):
        return self.value

    def create_nx_tree(self, string_tree: str) -> nx.DiGraph:
        nxTree = self.from_stringTree_to_nxTree(string_tree, self.grammars[0])
        return self.prune_tree(
            nxTree, terminal_to_torch_map_keys=self.terminal_to_op_names.keys()
        )

    def sample(self, user_priors: bool = False):  # pylint: disable=unused-argument
        self.reset()
        self.string_tree = self.grammars[0].sampler(1, user_priors=user_priors)[0]
        _ = self.value  # required for checking if graph is valid!

    @property
    def value(self):
        if self._value is None:
            if self.new_graph_repr_func:
                self._value = self.get_graph_representation(
                    self.id,
                    self.grammars[0],
                    edge_attr=self.edge_attr,
                )
            else:
                self._value = self.from_stringTree_to_graph_repr(
                    self.string_tree,
                    self.grammars[0],
                    valid_terminals=self.terminal_to_op_names.keys(),
                    edge_attr=self.edge_attr,
                )
        return self._value

    def create_from_id(self, identifier: str):
        self.reset()
        self.id = identifier
        self.string_tree = self.id_to_string_tree(self.id)
        _ = self.value  # required for checking if graph is valid!

    # TODO: does this serialization really work for every graph ?
    def serialize(self):
        return self.id

    def load_from(self, data):
        self.create_from_id(data)

    def mutate(
        self,
        parent: GraphGrammar = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ):
        if parent is None:
            parent = self
        parent_string_tree = parent.string_tree

        if mutation_strategy == "bananas":
            child_string_tree, is_same = bananas_mutate(
                parent_string_tree=parent_string_tree,
                grammar=self.grammars[0],
                mutation_rate=mutation_rate,
            )
        else:
            child_string_tree, is_same = simple_mutate(
                parent_string_tree=parent_string_tree,
                grammar=self.grammars[0],
            )

        if is_same:
            raise Exception("Parent is the same as child!")

        return parent.create_new_instance_from_id(
            self.string_tree_to_id(child_string_tree)
        )

    def crossover(self, parent1: GraphGrammar, parent2: GraphGrammar = None):
        if parent2 is None:
            parent2 = self
        parent1_string_tree = parent1.string_tree
        parent2_string_tree = parent2.string_tree
        children = simple_crossover(
            parent1_string_tree, parent2_string_tree, self.grammars[0]
        )
        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        return [
            parent2.create_new_instance_from_id(self.string_tree_to_id(child))
            for child in children
        ]

    def compute_prior(self, log: bool = True) -> float:
        return self.grammars[0].compute_prior(self.string_tree, log=log)


class GraphGrammarCell(GraphGrammar):
    hp_name = "graph_grammar_cell"

    def __init__(  # pylint: disable=W0102
        self,
        grammar: Grammar,
        terminal_to_op_names: dict,
        terminal_to_graph_edges: dict = None,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        name: str = None,
        scope: str = None,
        **kwargs,
    ):
        super().__init__(
            grammar,
            terminal_to_op_names,
            terminal_to_graph_edges,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
            **kwargs,
        )

        self.cell = None

    def reset(self):
        super().reset()
        self.cell = None

    @abstractmethod
    def create_graph_from_string(self, child: str):
        raise NotImplementedError


class GraphGrammarRepetitive(CoreGraphGrammar, Parameter):
    hp_name = "graph_grammar_repetitive"

    def __init__(  # pylint: disable=W0102
        self,
        grammars: list[Grammar],
        terminal_to_op_names: dict,
        terminal_to_sublanguage_map: dict,
        number_of_repetitive_motifs: int,
        terminal_to_graph_edges: dict = None,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        name: str = None,
        scope: str = None,
    ):
        CoreGraphGrammar.__init__(
            self,
            grammars=grammars,
            terminal_to_op_names=terminal_to_op_names,
            terminal_to_graph_edges=terminal_to_graph_edges,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
        )
        Parameter.__init__(self, set_default_value=False)

        self.id: str = ""
        self.string_tree: str = ""
        self.string_tree_list: list[str] = []
        self.nxTree: nx.DiGraph = None
        self._value: nx.DiGraph = None

        self.full_grammar = self.get_full_grammar(self.grammars)
        self.terminal_to_sublanguage_map = terminal_to_sublanguage_map
        self.number_of_repetitive_motifs = number_of_repetitive_motifs

    def __eq__(self, other):
        return self.id == other.id

    def reset(self):
        self.clear_graph()
        self.string_tree_list = []
        self.string_tree = ""
        self.nxTree = None
        self._value = None
        self.id = ""

    @staticmethod
    def get_full_grammar(grammars):
        full_grammar = deepcopy(grammars[0])
        rules = full_grammar.productions()
        nonterminals = full_grammar.nonterminals
        terminals = full_grammar.terminals
        for g in grammars[1:]:
            rules.extend(g.productions())
            nonterminals.extend(g.nonterminals)
            terminals.extend(g.terminals)
        return full_grammar

    @abstractmethod
    def create_graph_from_string(self, child: list[str]):
        raise NotImplementedError

    def get_dictionary(self) -> dict:
        return {"graph_grammar": "\n".join(self.string_tree_list)}

    def get_graphs(self):
        return self.value

    def create_nx_tree(self, string_tree: str) -> nx.DiGraph:
        nxTree = self.from_stringTree_to_nxTree(string_tree, self.full_grammar)
        return self.prune_tree(
            nxTree, terminal_to_torch_map_keys=self.terminal_to_op_names.keys()
        )

    def sample(self, user_priors: bool = False):  # pylint: disable=unused-argument
        self.reset()
        self.string_tree_list = [grammar.sampler(1)[0] for grammar in self.grammars]
        self.string_tree = self.assemble_trees(
            self.string_tree_list[0],
            self.string_tree_list[1:],
            terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
        )
        self.id = "\n".join(self.string_tree_list)
        _ = self.value  # required for checking if graph is valid!

    @property
    def value(self):
        if self._value is None:
            self._value = self.from_stringTree_to_graph_repr(
                self.string_tree,
                self.full_grammar,
                valid_terminals=self.terminal_to_op_names.keys(),
                edge_attr=self.edge_attr,
            )
        return self._value

    def create_from_id(self, identifier: str | list):
        self.reset()
        self.string_tree_list = (
            identifier.split("\n") if isinstance(identifier, str) else identifier
        )
        self.string_tree = self.assemble_trees(
            self.string_tree_list[0],
            self.string_tree_list[1:],
            terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
        )
        self.id = "\n".join(self.string_tree_list)
        _ = self.value  # required for checking if graph is valid!

    def mutate(
        self,
        parent: GraphGrammarRepetitive = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ):
        raise NotImplementedError
        # pylint: disable=unreachable
        if parent is None:
            parent = self

        # bananas mutate
        if mutation_strategy == "bananas":
            inner_mutation_strategy = partial(bananas_mutate, mutation_rate=mutation_rate)
            child_string_tree_list, is_same = repetitive_search_space_mutation(
                base_parent=parent.string_tree_list[0],
                motif_parents=parent.string_tree_list[1:],
                base_grammar=self.grammars[0],
                motif_grammars=self.grammars[1:],
                terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
                inner_mutation_strategy=inner_mutation_strategy,
            )
        else:
            child_string_tree_list, is_same = repetitive_search_space_mutation(
                base_parent=parent.string_tree_list[0],
                motif_parents=parent.string_tree_list[1:],
                base_grammar=self.grammars[0],
                motif_grammars=self.grammars[1:],
                terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
                inner_mutation_strategy=super().mutate,
            )

        if all(is_same):
            raise ValueError("Parent is the same as child!")

        return self.create_graph_from_string(child_string_tree_list)
        # pylint: enable=unreachable

    def crossover(
        self, parent1: GraphGrammarRepetitive, parent2: GraphGrammarRepetitive = None
    ):
        raise NotImplementedError
        # pylint: disable=unreachable
        if parent2 is None:
            parent2 = self
        children = repetitive_search_space_crossover(
            base_parent=(parent1.string_tree_list[0], parent2.string_tree_list[0]),
            motif_parents=(parent1.string_tree_list[1:], parent2.string_tree_list[1:]),
            base_grammar=self.grammars[0],
            motif_grammars=self.grammars[1:],
            terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
            inner_crossover_strategy=simple_crossover,
        )
        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        return [parent2.create_graph_from_string(child) for child in children]
        # pylint: enable=unreachable

    @property
    def search_space_size(self) -> int:
        def recursive_worker(
            nonterminal: Nonterminal, grammar, lower_level_motifs: int = 0
        ) -> int:
            primitive_nonterminal = "OPS"
            if str(nonterminal) == primitive_nonterminal:
                return (
                    lower_level_motifs * self.number_of_repetitive_motifs
                    + len(grammar.productions(lhs=Nonterminal(primitive_nonterminal)))
                    - self.number_of_repetitive_motifs
                )
            potential_productions = grammar.productions(lhs=nonterminal)
            _possibilites = 0
            for potential_production in potential_productions:
                edges_nonterminals = [
                    rhs_sym
                    for rhs_sym in potential_production.rhs()
                    if str(rhs_sym) in grammar.nonterminals
                ]
                possibilities_per_edge = [
                    recursive_worker(e_nonterminal, grammar, lower_level_motifs)
                    for e_nonterminal in edges_nonterminals
                ]
                product = 1
                for p in possibilities_per_edge:
                    product *= p
                _possibilites += product
            return _possibilites

        lower_level_motifs = recursive_worker(self.grammars[1].start(), self.grammars[1])
        return recursive_worker(
            self.grammars[0].start(),
            self.grammars[0],
            lower_level_motifs=lower_level_motifs,
        )


class GraphGrammarMultipleRepetitive(CoreGraphGrammar, Parameter):
    hp_name = "graph_grammar_multiple_repetitive"

    def __init__(  # pylint: disable=W0102
        self,
        grammars: list[Grammar] | list[ConstrainedGrammar],
        terminal_to_op_names: dict,
        terminal_to_sublanguage_map: dict,
        prior: list[dict] = None,
        terminal_to_graph_edges: dict = None,
        fixed_macro_grammar: bool = False,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        name: str = None,
        scope: str = None,
        **kwargs,
    ):
        def _check_mapping(macro_grammar, motif_grammars, terminal_to_sublanguage_map):
            for terminal, start_symbol in terminal_to_sublanguage_map.items():
                if terminal not in macro_grammar.terminals:
                    raise Exception(f"Terminal {terminal} not defined in macro grammar")
                if not any(
                    start_symbol == str(grammar.start()) for grammar in motif_grammars
                ):
                    raise Exception(
                        f"Start symbol {start_symbol} not defined in motif grammar"
                    )

        def _identify_macro_grammar(grammar, terminal_to_sublanguage_map):
            grammars = deepcopy(grammar)
            motif_grammars = []
            for start_symbol in terminal_to_sublanguage_map.values():
                motif_grammars += [
                    grammar
                    for grammar in grammars
                    if start_symbol == str(grammar.start())
                ]
                grammars = [
                    grammar
                    for grammar in grammars
                    if start_symbol != str(grammar.start())
                ]
            if len(grammars) != 1:
                raise Exception("Cannot identify macro grammar")
            return grammars[0], motif_grammars

        if prior is not None:
            assert len(grammars) == len(
                prior
            ), "At least one of the grammars has no prior defined!"
            for g, p in zip(grammars, prior):
                g.prior = p
        self.has_prior = prior is not None

        self.macro_grammar, grammars = _identify_macro_grammar(
            grammars, terminal_to_sublanguage_map
        )
        _check_mapping(self.macro_grammar, grammars, terminal_to_sublanguage_map)

        self.fixed_macro_grammar = fixed_macro_grammar
        if not self.fixed_macro_grammar:
            grammars.insert(0, self.macro_grammar)

        self.terminal_to_sublanguage_map = OrderedDict(terminal_to_sublanguage_map)
        if any(
            k in terminal_to_op_names for k in self.terminal_to_sublanguage_map.keys()
        ):
            raise Exception(
                f"Terminals {[k for k in self.terminal_to_sublanguage_map.keys()]} already defined in primitives mapping and cannot be used for repetitive substitutions"
            )
        self.number_of_repetitive_motifs_per_grammar = [
            sum(
                map(
                    (str(grammar.start())).__eq__,
                    self.terminal_to_sublanguage_map.values(),
                )
            )
            if str(grammar.start()) in self.terminal_to_sublanguage_map.values()
            else 1
            for grammar in grammars
        ]

        CoreGraphGrammar.__init__(
            self,
            grammars=grammars,
            terminal_to_op_names={
                **terminal_to_op_names,
                **self.terminal_to_sublanguage_map,
            },
            terminal_to_graph_edges=terminal_to_graph_edges,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
            **kwargs,
        )
        Parameter.__init__(self, set_default_value=False)

        self._function_id: str = ""
        self.string_tree: str = ""
        self.string_tree_list: list[str] = []
        self.nxTree: nx.DiGraph = None
        self._value: nx.DiGraph = None

        if self.fixed_macro_grammar:
            self.fixed_macro_string_tree = self.macro_grammar.sampler(1)[0]

        if self.fixed_macro_grammar:
            self.full_grammar = self.get_full_grammar(
                [self.macro_grammar] + self.grammars
            )
        else:
            self.full_grammar = self.get_full_grammar(self.grammars)

    @property
    def id(self) -> str:
        if self._function_id is None or self._function_id == "":
            if len(self.string_tree_list) == 0:
                raise ValueError("Cannot infer identifier")
            self._function_id = self.string_tree_list_to_id(self.string_tree_list)
        return self._function_id

    @id.setter
    def id(self, value: str):
        self._function_id = value

    @staticmethod
    def id_to_string_tree_list(identifier: str) -> list[str]:
        return identifier.split("\n")

    def id_to_string_tree(self, identifier: str) -> str:
        string_tree_list = self.id_to_string_tree_list(identifier)
        return self.assemble_string_tree(string_tree_list)

    @staticmethod
    def string_tree_list_to_id(string_tree_list: list[str]) -> str:
        return "\n".join(string_tree_list)

    def string_tree_to_id(self, string_tree: str) -> str:
        raise NotImplementedError

    def assemble_string_tree(self, string_tree_list: list[str]) -> str:
        if self.fixed_macro_grammar:
            string_tree = self.assemble_trees(
                self.fixed_macro_string_tree,
                string_tree_list,
                terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
            )
        else:
            string_tree = self.assemble_trees(
                string_tree_list[0],
                string_tree_list[1:],
                terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
            )
        return string_tree

    def __eq__(self, other):
        return self.id == other.id

    def reset(self):
        self.clear_graph()
        self.string_tree_list = []
        self.string_tree = ""
        self.nxTree = None
        self._value = None
        self._function_id = ""

    def compose_functions(self, flatten_graph: bool = True):
        return self._compose_functions(self.id, self.full_grammar, flatten_graph)

    def unparse_tree(self, identifier: str, as_composition: bool = True):
        return self._unparse_tree(identifier, self.full_grammar, as_composition)

    @staticmethod
    def get_full_grammar(grammars):
        full_grammar = deepcopy(grammars[0])
        rules = full_grammar.productions()
        nonterminals = full_grammar.nonterminals
        terminals = full_grammar.terminals
        for g in grammars[1:]:
            rules.extend(g.productions())
            nonterminals.extend(g.nonterminals)
            terminals.extend(g.terminals)
        return full_grammar

    def serialize(self):
        return self.id

    def load_from(self, data):
        self.create_from_id(data)

    @abstractmethod
    def create_new_instance_from_id(self, child: str):
        raise NotImplementedError

    def get_dictionary(self) -> dict:
        return {"graph_grammar": self.id}

    def get_graphs(self):
        return self.value

    def create_nx_tree(self, string_tree: str) -> nx.DiGraph:
        nxTree = self.from_stringTree_to_nxTree(string_tree, self.full_grammar)
        return self.prune_tree(
            nxTree, terminal_to_torch_map_keys=self.terminal_to_op_names.keys()
        )

    def sample(self, user_priors: bool = False):
        self.reset()
        self.string_tree_list = [
            grammar.sampler(1, user_priors=user_priors)[0]
            for grammar, number_of_motifs in zip(
                self.grammars, self.number_of_repetitive_motifs_per_grammar
            )
            for _ in range(number_of_motifs)
        ]
        self.string_tree = self.assemble_string_tree(self.string_tree_list)
        _ = self.value  # required for checking if graph is valid!

    @property
    def value(self):
        if self._value is None:
            if self.fixed_macro_grammar:
                self._value = []
                string_list_idx = 0
                for grammar, number_of_motifs in zip(
                    self.grammars, self.number_of_repetitive_motifs_per_grammar
                ):
                    for _ in range(number_of_motifs):
                        self._value.append(
                            self.from_stringTree_to_graph_repr(
                                self.string_tree_list[string_list_idx],
                                grammar,
                                valid_terminals=self.terminal_to_op_names.keys(),
                                edge_attr=self.edge_attr,
                            )
                        )
                        string_list_idx += 1
                self._value = self._value[0]  # TODO trick
            else:
                self._value = self.from_stringTree_to_graph_repr(
                    self.string_tree,
                    self.full_grammar,
                    valid_terminals=self.terminal_to_op_names.keys(),
                    edge_attr=self.edge_attr,
                )
                motif_trees = self.string_tree_list[1:]
                repetitive_mapping = {
                    replacement: motif
                    for motif, replacement in zip(
                        self.terminal_to_sublanguage_map.keys(), motif_trees
                    )
                }
                for subgraph in self._value[1].values():
                    old_node_attributes = nx.get_node_attributes(subgraph, "op_name")
                    new_node_labels = {
                        k: (repetitive_mapping[v] if v in motif_trees else v)
                        for k, v in old_node_attributes.items()
                    }
                    nx.set_node_attributes(subgraph, new_node_labels, name="op_name")
        return self._value

    def create_from_id(self, identifier: str):
        self.reset()
        self.id = identifier
        self.string_tree_list = self.id_to_string_tree_list(self.id)
        self.string_tree = self.id_to_string_tree(self.id)
        _ = self.value  # required for checking if graph is valid!

    def mutate(
        self,
        parent: GraphGrammarMultipleRepetitive = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ):
        if parent is None:
            parent = self

        bananas_inner_mutation = partial(bananas_mutate, mutation_rate=mutation_rate)
        child_string_tree_list, is_same = repetitive_search_space_mutation(
            base_parent=self.fixed_macro_string_tree
            if self.fixed_macro_grammar
            else parent.string_tree_list[0],
            motif_parents=parent.string_tree_list
            if self.fixed_macro_grammar
            else parent.string_tree_list[1:],
            base_grammar=self.macro_grammar,
            motif_grammars=self.grammars
            if self.fixed_macro_grammar
            else self.grammars[1:],
            terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
            number_of_repetitive_motifs_per_grammar=self.number_of_repetitive_motifs_per_grammar,
            inner_mutation_strategy=bananas_inner_mutation
            if mutation_strategy == "bananas"
            else super().mutate,
            fixed_macro_parent=self.fixed_macro_grammar,
        )

        if all(is_same):
            raise ValueError("Parent is the same as child!")

        if self.fixed_macro_grammar:
            child_string_tree_list = child_string_tree_list[1:]

        return self.create_new_instance_from_id(
            self.string_tree_list_to_id(child_string_tree_list)
        )

    def crossover(
        self,
        parent1: GraphGrammarMultipleRepetitive,
        parent2: GraphGrammarMultipleRepetitive = None,
    ):
        if parent2 is None:
            parent2 = self
        children = repetitive_search_space_crossover(
            base_parent=(parent1.fixed_macro_string_tree, parent2.fixed_macro_string_tree)
            if self.fixed_macro_grammar
            else (parent1.string_tree_list[0], parent2.string_tree_list[0]),
            motif_parents=(parent1.string_tree_list, parent2.string_tree_list)
            if self.fixed_macro_grammar
            else (parent1.string_tree_list[1:], parent2.string_tree_list[1:]),
            base_grammar=self.macro_grammar,
            motif_grammars=self.grammars
            if self.fixed_macro_grammar
            else self.grammars[1:],
            terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
            number_of_repetitive_motifs_per_grammar=self.number_of_repetitive_motifs_per_grammar,
            inner_crossover_strategy=simple_crossover,
            fixed_macro_parent=self.fixed_macro_grammar,
            multiple_repetitive=True,
        )
        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        return [
            parent2.create_new_instance_from_id(
                self.string_tree_list_to_id(
                    child[1:] if self.fixed_macro_grammar else child
                )
            )
            for child in children
        ]

    @property
    def search_space_size(self) -> int:
        def recursive_worker(
            nonterminal: Nonterminal, grammar, lower_level_motifs: dict = None
        ) -> int:
            if lower_level_motifs is None:
                lower_level_motifs = {}
            potential_productions = grammar.productions(lhs=nonterminal)
            _possibilites = 0
            for potential_production in potential_productions:
                edges_nonterminals = [
                    rhs_sym
                    for rhs_sym in potential_production.rhs()
                    if str(rhs_sym) in grammar.nonterminals
                ]
                possibilities_per_edge = [
                    recursive_worker(e_nonterminal, grammar, lower_level_motifs)
                    for e_nonterminal in edges_nonterminals
                ]
                possibilities_per_edge += [
                    lower_level_motifs[str(rhs_sym)]
                    for rhs_sym in potential_production.rhs()
                    if str(rhs_sym) in lower_level_motifs.keys()
                ]
                product = 1
                for p in possibilities_per_edge:
                    product *= p
                _possibilites += product
            return _possibilites

        if self.fixed_macro_grammar:
            if len(self.grammars) > 1:
                raise Exception(
                    "Compute space size for fixed macro only works for one repetitive level"
                )
            return np.prod(
                [
                    grammar.compute_space_size
                    for grammar, n_grammar in zip(
                        self.grammars, self.number_of_repetitive_motifs_per_grammar
                    )
                    for _ in range(n_grammar)
                ]
            )
        else:
            if len(self.grammars) > 2:
                raise Exception(
                    "Compute space size for no fixed macro only works for one repetitive level"
                )
            macro_space_size = self.grammars[0].compute_space_size
            motif_space_size = self.grammars[1].compute_space_size
            return (
                macro_space_size
                // self.number_of_repetitive_motifs_per_grammar[1]
                * motif_space_size
            )

    def compute_prior(self, log: bool = True) -> float:
        prior_probs = [
            g.compute_prior(st, log=log)
            for g, st in zip(self.grammars, self.string_tree_list)
        ]
        if log:
            return sum(prior_probs)
        else:
            return np.prod(prior_probs)
