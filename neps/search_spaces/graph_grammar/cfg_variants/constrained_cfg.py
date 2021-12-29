from collections import deque
from typing import Deque

from nltk.grammar import Nonterminal

from ..cfg import Grammar, choice


class ConstrainedGrammar(Grammar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints = None
        self.none_operation = None

    def set_constraints(self, constraints: dict, none_operation: str):
        self.constraints = constraints
        self.none_operation = none_operation

    @staticmethod
    def is_constrained():
        return True

    def sampler(
        self,
        n=1,
        start_symbol: str = None,
        not_allowed_productions=None,
    ):
        if start_symbol is None:
            start_symbol = self.start()
        else:
            start_symbol = Nonterminal(start_symbol)

        return [
            self._constrained_sampler(
                symbol=start_symbol,
                not_allowed_productions=not_allowed_productions,
            )
            + ")"
            for _ in range(n)
        ]

    def _get_not_allowed_productions(
        self, potential_productions, any_production_allowed: bool
    ):
        return list(
            filter(
                lambda prod: not any_production_allowed
                and len(prod.rhs()) == 1
                and prod.rhs()[0] == self.none_operation,
                potential_productions,
            )
        )

    def _constrained_sampler(
        self,
        symbol=None,
        not_allowed_productions=None,
        current_derivation=None,
    ):
        # simple sampler where each production is sampled uniformly from all possible productions
        # Tree choses if return tree or list of terminals
        # recursive implementation

        # init the sequence
        tree = "(" + str(symbol)
        # collect possible productions from the starting symbol
        productions = self.productions(lhs=symbol)
        if not_allowed_productions is not None:
            productions = list(
                filter(
                    lambda production: production not in not_allowed_productions,
                    productions,
                )
            )
        # sample
        production = choice(productions)
        counter = 0
        current_derivation = self.constraints(production.rhs()[0])
        for sym in production.rhs():
            if isinstance(sym, str):
                # if terminal then add string to sequence
                tree = tree + " " + sym
            else:
                context_information = self.constraints(
                    production.rhs()[0],
                    current_derivation,
                )
                not_allowed_productions = self._get_not_allowed_productions(
                    self.productions(lhs=sym), context_information[counter]
                )
                ret_val = self._constrained_sampler(sym, not_allowed_productions)
                tree = tree + " " + ret_val + ")"
                current_derivation[counter] = ret_val
                counter += 1
        return tree

    def _compute_current_context(self, pre_subtree: str, post_subtree: str):
        q_nonterminals: Deque = deque()
        for sym in pre_subtree.split(" "):
            if sym == "":
                continue
            if sym[0] == "(":
                sym = sym[1:]
            if sym[-1] == ")":
                for _ in range(sym.count(")")):
                    q_nonterminals.pop()
                while sym[-1] == ")":
                    sym = sym[:-1]
            if len(sym) == 1 and sym[0] in [" ", "\t", "\n", "[", "]"]:
                continue
            if sym in self.nonterminals:
                q_nonterminals.append(sym)

        context_start_idx = pre_subtree.rfind(q_nonterminals[-1])
        pre_subtree_context = pre_subtree[context_start_idx:]
        topology = pre_subtree_context[len(q_nonterminals[-1]) + 1 :].split(" ")[0]
        productions = [
            prod
            for prod in self.productions()
            if pre_subtree_context[: len(q_nonterminals[-1])] == f"{prod.lhs()}"
            and prod.rhs()[0] == topology
        ]
        if len(productions) == 0:
            raise Exception("Cannot find corresponding production!")
        if len(productions) > 1:
            raise NotImplementedError
        production = productions[0]

        q_context: Deque = deque()
        current_derivation = self.constraints(production.rhs()[0])
        counter = 0
        pre_subtree_context = pre_subtree_context[len(q_nonterminals[-1]) + 1 :]
        for s in pre_subtree_context.split(" "):
            if s == "":
                continue
            if s[0] == "(":
                q_context.append(s)
            elif s[-1] == ")":
                while s[-1] == ")":
                    q_context.pop()
                    s = s[:-1]
                if len(q_context) == 0:
                    current_derivation[counter] = s
                    counter += 1
                if counter == len(current_derivation):
                    break

        counter += 1
        if counter < len(current_derivation):
            q_context = deque()
            for s in post_subtree.split(" "):
                if s == "":
                    continue
                elif s[0] == "(":
                    q_context.append(s)
                elif s[-1] == ")":
                    while s[-1] == ")":
                        q_context.pop()
                        s = s[:-1]
                    if len(q_context) == 0:
                        current_derivation[counter] = s
                        counter += 1
                    if counter == len(current_derivation):
                        break

        return topology, current_derivation

    def mutate(
        self, parent: str, subtree_index: int, subtree_node: str, patience: int = 50
    ):
        # chop out subtree
        pre, _, post = self.remove_subtree(parent, subtree_index)
        rhs, current_derivation = self._compute_current_context(pre, post)
        context_information = self.constraints(
            rhs,
            current_derivation,
        )
        not_allowed_productions = self._get_not_allowed_productions(
            self.productions(lhs=Nonterminal(subtree_node)),
            context_information[
                [i for i, cd in enumerate(current_derivation) if cd is None][0]
            ],
        )
        _patience = patience
        while _patience > 0:
            # only sample subtree -> avoids full sampling of large parse trees
            new_subtree = self.sampler(
                1,
                start_symbol=subtree_node,
                not_allowed_productions=not_allowed_productions,
            )[0]
            child = pre + new_subtree + post
            if parent != child:  # ensure that parent is really mutated
                break
            _patience -= 1
        return child

    def crossover(self, parent1: str, parent2: str, patience: int = 50):
        subtree_node, subtree_index = self.rand_subtree(parent1)
        # chop out subtree
        pre, sub, post = self.remove_subtree(parent1, subtree_index)
        rhs, current_derivation = self._compute_current_context(pre, post)
        context_information = self.constraints(
            rhs,
            current_derivation,
        )
        parent1_not_allowed_productions = self._get_not_allowed_productions(
            self.productions(lhs=Nonterminal(subtree_node)),
            context_information[
                [i for i, cd in enumerate(current_derivation) if cd is None][0]
            ],
        )
        _patience = patience
        while _patience > 0:
            # sample subtree from donor
            donor_subtree_index = self.rand_subtree_fixed_head(parent2, subtree_node)
            # if no subtrees with right head node return False
            if not donor_subtree_index:
                _patience -= 1
            else:
                donor_pre, donor_sub, donor_post = self.remove_subtree(
                    parent2, donor_subtree_index
                )
                rhs, current_derivation = self._compute_current_context(
                    donor_pre, donor_post
                )
                context_information = self.constraints(
                    rhs,
                    current_derivation,
                )
                parent2_not_allowed_productions = self._get_not_allowed_productions(
                    self.productions(lhs=Nonterminal(subtree_node)),
                    context_information[
                        [i for i, cd in enumerate(current_derivation) if cd is None][0]
                    ],
                )
                if (
                    parent1_not_allowed_productions is not None
                    and "zero" in donor_sub
                    and donor_sub.count("(") == 1
                    and donor_sub.count(")") == 1
                ):
                    _patience -= 1
                    continue
                if (
                    parent2_not_allowed_productions is not None
                    and "zero" in sub
                    and sub.count("(") == 1
                    and sub.count(")") == 1
                ):
                    _patience -= 1
                    continue
                # return the two new tree
                child1 = pre + donor_sub + post
                child2 = donor_pre + sub + donor_post
                return child1, child2

        return False, False
