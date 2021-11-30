from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from functools import partial

import networkx as nx
from nltk import Nonterminal

from ..hyperparameter import Hyperparameter
from .cfg import Grammar
from .core_graph_grammar import CoreGraphGrammar
from .crossover import repetitive_search_space_crossover, simple_crossover
from .mutations import bananas_mutate, repetitive_search_space_mutation


class GraphGrammar(CoreGraphGrammar, Hyperparameter):
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

        self.id: str = None
        self.string_tree: str = ""
        self.nxTree: nx.DiGraph = None
        self.value: nx.DiGraph = None

    def get_search_space_size(self, primitive_nonterminal: str = "OPS") -> int:
        if len(self.grammars) != 1:
            raise NotImplementedError("Does not support multiple grammars")
        return self.grammars[0].compute_space_size(
            primitive_nonterminal=primitive_nonterminal
        )

    def reset(self):
        self.clear_graph()
        self.string_tree = ""
        self.nxTree = None
        self.value = None
        self.id = None

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

    def sample(self):
        self.reset()
        self.string_tree = self.grammars[0].sampler(1)[0]
        self.id = self.string_tree
        self.create_representation(self.string_tree)

    def create_representation(self, string_tree: str):
        self.value = (
            self.create_nx_tree(string_tree)
            if self.id_parse_tree
            else self.from_stringTree_to_graph_repr(
                string_tree,
                self.grammars[0],
                terminal_to_graph_edges=self.terminal_to_graph_repr,
                valid_terminals=self.terminal_to_op_names.keys(),
                edge_attr=self.edge_attr,
            )
        )

    def create_from_id(self, identifier: str):
        self.id = identifier
        self.string_tree = self.id
        self.create_representation(self.string_tree)

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
            child_string_tree, is_same = super().mutate(
                parent_string_tree=parent_string_tree
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

    def _get_neighbours(self, **kwargs):
        pass

    def _inv_transform(self):
        pass

    def _transform(self):
        pass


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


class GraphGrammarRepetitive(CoreGraphGrammar, Hyperparameter):
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

        self.id: str = None
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
        self.id = None

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
    def create_graph_from_string(self, child: str):
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
        self.string_tree_list: list[str] = [
            grammar.sampler(1)[0] for grammar in self.grammars
        ]
        self.string_tree = self.assemble_trees(
            self.string_tree_list[0],
            self.string_tree_list[1:],
            base_to_motif_map=self.base_to_motif_map,
        )
        self.id = self.string_tree
        self.create_representation(self.string_tree)

    def create_representation(self, string_tree: str):
        self.value = (
            self.create_nx_tree(string_tree)
            if self.id_parse_tree
            else self.from_stringTree_to_graph_repr(
                string_tree,
                self.grammars[0],
                terminal_to_graph_edges=self.terminal_to_graph_repr,
                valid_terminals=self.terminal_to_op_names.keys(),
                edge_attr=self.edge_attr,
            )
        )

    def create_from_id(self, identifier: str):
        self.string_tree_list = identifier.split("\n")
        self.string_tree = self.assemble_trees(
            self.string_tree_list[0],
            self.string_tree_list[1:],
            base_to_motif_map=self.base_to_motif_map,
        )
        self.id = self.string_tree
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
            mutation_strategy = partial(bananas_mutate, mutation_rate=mutation_rate)
            child_string_tree_list, is_same = repetitive_search_space_mutation(
                base_parent=parent.string_tree_list[0],
                motif_parents=parent.string_tree_list[1:],
                base_grammar=self.grammars[0],
                motif_grammars=self.grammars[1:],
                inner_mutation_strategy=mutation_strategy,
            )
        else:
            child_string_tree_list, is_same = repetitive_search_space_mutation(
                base_parent=parent.string_tree[0],
                motif_parents=parent.string_tree[1:],
                base_grammar=self.grammars[0],
                motif_grammars=self.grammars[1:],
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
            inner_crossover_strategy=simple_crossover,
        )
        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        return [parent2.create_graph_from_string(child) for child in children]

    def get_search_space_size(self, primitive_nonterminal: str = "OPS") -> int:
        def recursive_worker(
            nonterminal: Nonterminal, grammar, lower_level_motifs: int = 0
        ) -> int:
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

    def _get_neighbours(self, **kwargs):
        pass

    def _inv_transform(self):
        pass

    def _transform(self):
        pass
