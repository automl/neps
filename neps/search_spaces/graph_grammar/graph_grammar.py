from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from functools import partial

import networkx as nx
import numpy as np
from nltk import Nonterminal

from ..parameter import Parameter
from .cfg import Grammar
from .core_graph_grammar import CoreGraphGrammar
from .crossover import repetitive_search_space_crossover, simple_crossover
from .mutations import bananas_mutate, repetitive_search_space_mutation, simple_mutate


class GraphGrammar(CoreGraphGrammar, Parameter):
    hp_name = "graph_grammar"

    def __init__(  # pylint: disable=W0102
        self,
        grammar: Grammar,
        terminal_to_op_names: dict,
        terminal_to_graph_repr: dict,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        name: str = None,
        scope: str = None,
        id_parse_tree: bool = False,
    ):
        super().__init__(
            grammars=[grammar],
            terminal_to_op_names=terminal_to_op_names,
            terminal_to_graph_repr=terminal_to_graph_repr,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
        )

        self.id_parse_tree = id_parse_tree

        self.id: str = ""
        self.string_tree: str = ""
        self.nxTree: nx.DiGraph = None
        self.value: nx.DiGraph = None

        self.has_prior = False

    @property
    def search_space_size(self) -> int:
        if len(self.grammars) != 1:
            raise NotImplementedError("Does not support multiple grammars")
        return self.grammars[0].compute_space_size

    def reset(self):
        self.clear_graph()
        self.string_tree = ""
        self.nxTree = None
        self.value = None
        self.id = ""

    @abstractmethod
    def create_graph_from_string(self, child: str):
        raise NotImplementedError

    @abstractmethod
    def setup(self, tree: nx.DiGraph):
        raise NotImplementedError

    def get_dictionary(self) -> dict:
        return {"graph_grammar": self.id}

    def get_graphs(self):
        return self.value

    def create_nx_tree(self, string_tree: str) -> nx.DiGraph:
        nxTree = self.from_stringTree_to_nxTree(string_tree, self.grammars[0])
        return self.prune_tree(
            nxTree, terminal_to_torch_map_keys=self.terminal_to_op_names.keys()
        )

    def sample(self, use_user_priors: bool = False):  # pylint: disable=unused-argument
        self.reset()
        self.string_tree = self.grammars[0].sampler(1)[0]
        self.id = self.string_tree
        self.create_representation(self.string_tree)

    def create_representation(
        self, string_tree: str
    ):  # todo relevant code for conversion
        self.value = (
            self.create_nx_tree(string_tree)
            if self.id_parse_tree
            else self.from_stringTree_to_graph_repr(
                string_tree,
                self.grammars[0],
                terminal_to_graph_edges=self.terminal_to_graph_repr,
                valid_terminals=self.terminal_to_op_names.keys(),
                edge_attr=self.edge_attr,  # set to false for node attribute
            )
        )

    def create_from_id(self, identifier: str):
        self.id = identifier
        self.string_tree = self.id
        self.create_representation(self.string_tree)  # todo relevant for conversion

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

        return parent.create_graph_from_string(child_string_tree)

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
        return [parent2.create_graph_from_string(child) for child in children]


class GraphGrammarCell(GraphGrammar):
    hp_name = "graph_grammar_cell"

    def __init__(  # pylint: disable=W0102
        self,
        grammar: Grammar,
        terminal_to_op_names: dict,
        terminal_to_graph_repr: dict,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        name: str = None,
        scope: str = None,
        id_parse_tree: bool = False,
    ):
        super().__init__(
            grammar,
            terminal_to_op_names,
            terminal_to_graph_repr,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
            id_parse_tree=id_parse_tree,
        )

        self.cell = None

    def reset(self):
        super().reset()
        self.cell = None

    @abstractmethod
    def create_graph_from_string(self, child: str):
        raise NotImplementedError

    @abstractmethod
    def setup(self, tree: nx.DiGraph):
        raise NotImplementedError


class GraphGrammarRepetitive(CoreGraphGrammar, Parameter):
    hp_name = "graph_grammar_repetitive"

    def __init__(  # pylint: disable=W0102
        self,
        grammars: list[Grammar],
        terminal_to_op_names: dict,
        terminal_to_graph_repr: dict,
        base_to_motif_map: dict,
        number_of_repetitive_motifs: int,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        name: str = None,
        scope: str = None,
        id_parse_tree: bool = False,
    ):
        super().__init__(
            grammars=grammars,
            terminal_to_op_names=terminal_to_op_names,
            terminal_to_graph_repr=terminal_to_graph_repr,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
        )

        self.id_parse_tree = id_parse_tree

        self.id: str = ""
        self.string_tree: str = ""
        self.string_tree_list: list[str] = []
        self.nxTree: nx.DiGraph = None
        self.value: nx.DiGraph = None

        self.full_grammar = self.get_full_grammar(self.grammars)
        self.base_to_motif_map = base_to_motif_map
        self.number_of_repetitive_motifs = number_of_repetitive_motifs

    def reset(self):
        self.clear_graph()
        self.string_tree_list = []
        self.string_tree = ""
        self.nxTree = None
        self.value = None
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

    @abstractmethod
    def setup(self, tree: nx.DiGraph):
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

    def sample(self):
        self.reset()
        self.string_tree_list = [grammar.sampler(1)[0] for grammar in self.grammars]
        self.string_tree = self.assemble_trees(
            self.string_tree_list[0],
            self.string_tree_list[1:],
            base_to_motif_map=self.base_to_motif_map,
        )
        self.id = "\n".join(self.string_tree_list)
        self.create_representation(self.string_tree)

    def create_representation(self, string_tree: str):
        self.value = (
            self.create_nx_tree(string_tree)
            if self.id_parse_tree
            else self.from_stringTree_to_graph_repr(
                string_tree,
                self.full_grammar,
                terminal_to_graph_edges=self.terminal_to_graph_repr,
                valid_terminals=self.terminal_to_op_names.keys(),
                edge_attr=self.edge_attr,
            )
        )

    def create_from_id(self, identifier: str | list):
        self.string_tree_list = (
            identifier.split("\n") if isinstance(identifier, str) else identifier
        )
        self.string_tree = self.assemble_trees(
            self.string_tree_list[0],
            self.string_tree_list[1:],
            base_to_motif_map=self.base_to_motif_map,
        )
        self.id = "\n".join(self.string_tree_list)
        self.create_representation(self.string_tree)

    def mutate(
        self,
        parent: GraphGrammarRepetitive = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ):
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
                base_to_motif_map=self.base_to_motif_map,
                inner_mutation_strategy=inner_mutation_strategy,
            )
        else:
            child_string_tree_list, is_same = repetitive_search_space_mutation(
                base_parent=parent.string_tree_list[0],
                motif_parents=parent.string_tree_list[1:],
                base_grammar=self.grammars[0],
                motif_grammars=self.grammars[1:],
                base_to_motif_map=self.base_to_motif_map,
                inner_mutation_strategy=super().mutate,
            )

        if all(is_same):
            raise ValueError("Parent is the same as child!")

        return self.create_graph_from_string(child_string_tree_list)

    def crossover(
        self, parent1: GraphGrammarRepetitive, parent2: GraphGrammarRepetitive = None
    ):
        if parent2 is None:
            parent2 = self
        children = repetitive_search_space_crossover(
            base_parent=(parent1.string_tree_list[0], parent2.string_tree_list[0]),
            motif_parents=(parent1.string_tree_list[1:], parent2.string_tree_list[1:]),
            base_grammar=self.grammars[0],
            motif_grammars=self.grammars[1:],
            base_to_motif_map=self.base_to_motif_map,
            inner_crossover_strategy=simple_crossover,
        )
        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        return [parent2.create_graph_from_string(child) for child in children]

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
        macro_grammar: Grammar,
        grammars: list[Grammar],
        terminal_to_op_names: dict,
        terminal_to_graph_repr: dict,
        base_to_motif_map: dict,
        number_of_repetitive_motifs: list[int],
        fixed_macro_grammar: bool = False,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        name: str = None,
        scope: str = None,
        id_parse_tree: bool = False,
    ):
        self.macro_grammar = macro_grammar
        self.fixed_macro_grammar = fixed_macro_grammar
        if not self.fixed_macro_grammar:
            grammars.insert(0, macro_grammar)
        super().__init__(
            grammars=grammars,
            terminal_to_op_names=terminal_to_op_names,
            terminal_to_graph_repr=terminal_to_graph_repr,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
        )

        self.id_parse_tree = id_parse_tree

        self.id: str = ""
        self.string_tree: str = ""
        self.string_tree_list: list[str] = []
        self.nxTree: nx.DiGraph = None
        self.value: nx.DiGraph = None

        if self.fixed_macro_grammar:
            self.full_grammar = self.get_full_grammar([macro_grammar] + self.grammars)
        else:
            self.full_grammar = self.get_full_grammar(self.grammars)

        if self.fixed_macro_grammar:
            self.fixed_macro_string_tree = self.macro_grammar.sampler(1)[0]

        self.base_to_motif_map = base_to_motif_map
        self.number_of_repetitive_motifs = number_of_repetitive_motifs

    def reset(self):
        self.clear_graph()
        self.string_tree_list = []
        self.string_tree = ""
        self.nxTree = None
        self.value = None
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

    @abstractmethod
    def setup(self, tree: nx.DiGraph):
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

    def sample(self):
        self.reset()
        self.string_tree_list = [grammar.sampler(1)[0] for grammar in self.grammars]
        self.id = "\n".join(self.string_tree_list)
        if self.fixed_macro_grammar:
            self.string_tree = self.assemble_trees(
                self.fixed_macro_string_tree,
                self.string_tree_list,
                base_to_motif_map=self.base_to_motif_map,
            )
            self.create_representation(self.string_tree_list)
        else:
            self.string_tree = self.assemble_trees(
                self.string_tree_list[0],
                self.string_tree_list[1:],
                base_to_motif_map=self.base_to_motif_map,
            )
            self.create_representation(self.string_tree)

    def create_representation(self, string_tree: str | list[str]):
        if isinstance(string_tree, str):
            self.value = (
                self.create_nx_tree(string_tree)
                if self.id_parse_tree
                else self.from_stringTree_to_graph_repr(
                    string_tree,
                    self.full_grammar,
                    terminal_to_graph_edges=self.terminal_to_graph_repr,
                    valid_terminals=self.terminal_to_op_names.keys(),
                    edge_attr=self.edge_attr,
                )
            )
        elif isinstance(string_tree, list):
            self.value = []
            for g, st in zip(self.grammars, string_tree):
                self.value.append(
                    self.create_nx_tree(st)
                    if self.id_parse_tree
                    else self.from_stringTree_to_graph_repr(
                        st,
                        g,
                        terminal_to_graph_edges=self.terminal_to_graph_repr,
                        valid_terminals=self.terminal_to_op_names.keys(),
                        edge_attr=self.edge_attr,
                    )
                )
        else:
            raise NotImplementedError

    def create_from_id(self, identifier: str | list):
        self.string_tree_list = (
            identifier.split("\n") if isinstance(identifier, str) else identifier
        )
        self.id = "\n".join(self.string_tree_list)
        if self.fixed_macro_grammar:
            self.string_tree = self.assemble_trees(
                self.fixed_macro_string_tree,
                self.string_tree_list,
                base_to_motif_map=self.base_to_motif_map,
            )
            self.create_representation(self.string_tree_list)
        else:
            self.string_tree = self.assemble_trees(
                self.string_tree_list[0],
                self.string_tree_list[1:],
                base_to_motif_map=self.base_to_motif_map,
            )
            self.create_representation(self.string_tree)

    def mutate(
        self,
        parent: GraphGrammarMultipleRepetitive = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ):
        if parent is None:
            parent = self

        # bananas mutate
        if mutation_strategy == "bananas":
            inner_mutation_strategy = partial(bananas_mutate, mutation_rate=mutation_rate)
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
                base_to_motif_map=self.base_to_motif_map,
                inner_mutation_strategy=inner_mutation_strategy,
                fixed_macro_parent=self.fixed_macro_grammar,
            )
        else:
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
                base_to_motif_map=self.base_to_motif_map,
                inner_mutation_strategy=super().mutate,
                fixed_macro_parent=self.fixed_macro_grammar,
            )

        if all(is_same):
            raise ValueError("Parent is the same as child!")

        if self.fixed_macro_grammar:
            child_string_tree_list = child_string_tree_list[1:]

        return self.create_graph_from_string(child_string_tree_list)

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
            base_to_motif_map=self.base_to_motif_map,
            inner_crossover_strategy=simple_crossover,
            fixed_macro_parent=self.fixed_macro_grammar,
            multiple_repetitive=True,
        )
        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        return [
            parent2.create_graph_from_string(
                child[1:] if self.fixed_macro_grammar else child
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
            return np.prod(
                [grammar.compute_space_size for grammar in self.grammars]
                # [recursive_worker(grammar.start(), grammar) for grammar in self.grammars]
            )
        else:
            lower_level_motifs = {
                k: recursive_worker(self.grammars[i + 1].start(), self.grammars[i + 1])
                for i, k in enumerate(self.base_to_motif_map.keys())
            }
            macro_level_motifs = recursive_worker(
                self.grammars[0].start(),
                self.grammars[0],
                lower_level_motifs=lower_level_motifs,
            )
            return np.prod(list(lower_level_motifs.values()) + [macro_level_motifs])
