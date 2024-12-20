from __future__ import annotations

import math
from collections import deque
from functools import partial
from queue import LifoQueue

import numpy as np
from nltk.grammar import Nonterminal

from neps.search_spaces.architecture.cfg import Grammar


class Constraint:
    def __init__(self, current_derivation: str | None = None) -> None:
        self.current_derivation = current_derivation

    @staticmethod
    def initialize_constraints(topology: str) -> Constraint:
        raise NotImplementedError

    def get_not_allowed_productions(self, productions: str) -> list[bool] | bool:
        raise NotImplementedError

    def update_context(self, new_part: str) -> None:
        raise NotImplementedError

    def mutate_not_allowed_productions(
        self, nonterminal: str, before: str, after: str, possible_productions: list
    ) -> list:
        raise NotImplementedError


class ConstrainedGrammar(Grammar):
    def set_constraints(self, constraints: dict, none_operation: str | None = None):
        self.constraints = constraints
        self.none_operation = none_operation
        self.constraint_is_class = isinstance(self.constraints, Constraint)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints = None
        self.none_operation = None
        self.constraint_is_class: bool = False

        self._prior: dict | None = None

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value: dict):
        def _check_prior(value: dict):
            for nonterminal in self.nonterminals:
                if nonterminal not in value:
                    raise Exception(
                        f"Nonterminal {nonterminal} not defined in prior distribution!"
                    )
                if len(value[nonterminal]) != len(
                    self.productions(lhs=Nonterminal(nonterminal))
                ):
                    raise Exception(
                        f"Not all RHS of nonterminal {nonterminal} have a probability!"
                    )
                if not math.isclose(sum(value[nonterminal]), 1.0):
                    raise Exception(
                        f"Prior for {nonterminal} is no probablility distribution (=1)!"
                    )

        if value is not None:
            _check_prior(value)
        self._prior = value

    def sampler(  # type: ignore[override]
        self,
        n=1,
        start_symbol: str | None = None,
        not_allowed_productions=None,
        user_priors: bool = False,
    ):
        start_symbol = self.start() if start_symbol is None else Nonterminal(start_symbol)

        return [
            self._constrained_sampler(
                symbol=start_symbol,
                not_allowed_productions=not_allowed_productions,
                user_priors=user_priors,
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
        user_priors: bool = False,
    ):
        # simple sampler where each production is sampled uniformly from all possible productions
        # Tree choses if return tree or list of terminals
        # recursive implementation

        # init the sequence
        tree = "(" + str(symbol)
        # collect possible productions from the starting symbol
        productions = list(self.productions(lhs=symbol))
        if not_allowed_productions is not None and len(not_allowed_productions) > 0:
            productions = list(
                filter(
                    lambda production: production not in not_allowed_productions,
                    productions,
                )
            )

        if len(productions) == 0:
            raise Exception(f"There is no production possible for {symbol}")

        # sample
        if user_priors and self._prior is not None:
            probs = self._prior[str(symbol)]
            if not_allowed_productions:
                # remove prior probs if rule is not allowed
                not_allowed_indices = [
                    self.productions(lhs=symbol).index(nap)
                    for nap in not_allowed_productions
                ]
                probs = [p for i, p in enumerate(probs) if i not in not_allowed_indices]
                # rescale s.t. probs sum up to one
                cur_prob_sum = sum(probs)
                probs = [x / cur_prob_sum for x in probs]
            assert len(probs) == len(productions)

            production = np.random.choice(productions, p=probs)
        else:
            production = np.random.choice(productions)
        counter = 0
        if self.constraint_is_class:
            constraints = self.constraints.initialize_constraints(production.rhs()[0])
        else:
            current_derivation = self.constraints(production.rhs()[0])
        for sym in production.rhs():
            if isinstance(sym, str):
                # if terminal then add string to sequence
                tree = tree + " " + sym
            else:
                if self.constraint_is_class:
                    not_allowed_productions = (
                        constraints.get_not_allowed_productions(
                            productions=self.productions(lhs=sym)
                        )
                        if constraints is not None
                        else []
                    )
                else:
                    context_information = self.constraints(
                        production.rhs()[0],
                        current_derivation,
                    )
                    if isinstance(context_information, list):
                        not_allowed_productions = self._get_not_allowed_productions(
                            self.productions(lhs=sym), context_information[counter]
                        )
                    elif isinstance(context_information, bool):
                        not_allowed_productions = self._get_not_allowed_productions(
                            self.productions(lhs=sym), context_information
                        )
                    else:
                        raise NotImplementedError
                ret_val = self._constrained_sampler(
                    sym, not_allowed_productions, user_priors=user_priors
                )
                tree = tree + " " + ret_val + ")"
                if self.constraint_is_class:
                    constraints.update_context(ret_val + ")")
                else:
                    current_derivation[counter] = ret_val + ")"
                counter += 1
        return tree

    def compute_prior(self, string_tree: str, log: bool = True) -> float:
        def skip_char(char: str) -> bool:
            if char in [" ", "\t", "\n"]:
                return True
            # special case: "(" is (part of) a terminal
            if (
                i != 0
                and char == "("
                and string_tree[i - 1] == " "
                and string_tree[i + 1] == " "
            ):
                return False
            return char == "("

        def find_longest_match(
            i: int, string_tree: str, symbols: list, max_match: int
        ) -> int:
            # search for longest matching symbol and add it
            # assumes that the longest match is the true match
            j = min(i + max_match, len(string_tree) - 1)
            while j > i and j < len(string_tree):
                if string_tree[i:j] in symbols:
                    break
                j -= 1
            if j == i:
                raise Exception(f"Terminal or nonterminal at position {i} does not exist")
            return j

        prior_prob = 1.0 if not log else 0.0

        symbols = self.nonterminals + self.terminals
        max_match = max(map(len, symbols))
        find_longest_match_func = partial(
            find_longest_match,
            string_tree=string_tree,
            symbols=symbols,
            max_match=max_match,
        )

        q_production_rules: LifoQueue = LifoQueue()
        current_derivations = {}

        i = 0
        while i < len(string_tree):
            char = string_tree[i]
            if skip_char(char):
                pass
            elif char == ")" and string_tree[i - 1] != " ":
                # closing symbol of production
                production = q_production_rules.get(block=False)[0][0]
                idx = self.productions(production.lhs()).index(production)
                prior = self.prior[str(production.lhs())]
                if any(
                    prod.rhs()[0] == self.none_operation
                    for prod in self.productions(production.lhs())
                ):
                    outer_production = q_production_rules.queue[-1][0][0]
                    if len(q_production_rules.queue) not in current_derivations:
                        current_derivations[len(q_production_rules.queue)] = (
                            self.constraints(outer_production.rhs()[0])
                        )
                    context_information = self.constraints(
                        outer_production.rhs()[0],
                        current_derivations[len(q_production_rules.queue)],
                    )
                    if isinstance(context_information, list):
                        not_allowed_productions = self._get_not_allowed_productions(
                            self.productions(lhs=production.lhs()),
                            context_information[
                                current_derivations[len(q_production_rules.queue)].index(
                                    None
                                )
                            ],
                        )
                    elif isinstance(context_information, bool):
                        not_allowed_productions = self._get_not_allowed_productions(
                            self.productions(lhs=production.lhs()),
                            context_information,
                        )
                    else:
                        raise NotImplementedError
                    current_derivations[len(q_production_rules.queue)][
                        current_derivations[len(q_production_rules.queue)].index(None)
                    ] = production.rhs()[0]
                    if None not in current_derivations[len(q_production_rules.queue)]:
                        del current_derivations[len(q_production_rules.queue)]
                    if len(not_allowed_productions) > 0:
                        # remove prior prior if rule is not allowed
                        not_allowed_indices = [
                            self.productions(lhs=production.lhs()).index(nap)
                            for nap in not_allowed_productions
                        ]
                        prior = [
                            p for i, p in enumerate(prior) if i not in not_allowed_indices
                        ]
                        # rescale s.t. prior sum up to one
                        cur_prob_sum = sum(prior)
                        prior = [x / cur_prob_sum for x in prior]
                        idx -= sum(idx > i for i in not_allowed_indices)

                prior = prior[idx]
                if log:
                    prior_prob += np.log(prior + 1e-12)
                else:
                    prior_prob *= prior
            else:
                j = find_longest_match_func(i)
                sym = string_tree[i:j]
                i = j - 1

                if sym in self.terminals:
                    q_production_rules.queue[-1][0] = [
                        production
                        for production in q_production_rules.queue[-1][0]
                        if production.rhs()[q_production_rules.queue[-1][1]] == sym
                    ]
                    q_production_rules.queue[-1][1] += 1
                elif sym in self.nonterminals:
                    if not q_production_rules.empty():
                        q_production_rules.queue[-1][0] = [
                            production
                            for production in q_production_rules.queue[-1][0]
                            if str(production.rhs()[q_production_rules.queue[-1][1]])
                            == sym
                        ]
                        q_production_rules.queue[-1][1] += 1
                    q_production_rules.put([self.productions(lhs=Nonterminal(sym)), 0])
                else:
                    raise Exception(f"Unknown symbol {sym}")
            i += 1

        if not q_production_rules.empty():
            raise Exception(f"Error in prior computation for {string_tree}")

        return prior_prob

    def _compute_current_context(self, pre_subtree: str, post_subtree: str):
        q_nonterminals: deque = deque()
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

        q_context: deque = deque()
        current_derivation = []
        rhs_counter = 0
        tmp_str = ""
        for s in pre_subtree_context[len(q_nonterminals[-1]) + 1 :].split(" "):
            if s == "":
                continue
            if s[0] == "(":
                if len(q_context) == 0 and len(s) > 1:
                    productions = [
                        production
                        for production in productions
                        if [
                            str(prod_sym)
                            for prod_sym in production.rhs()
                            if isinstance(prod_sym, Nonterminal)
                        ][rhs_counter]
                        == s[1:]
                    ]
                    rhs_counter += 1
                q_context.append(s)
                tmp_str += " " + s
            elif s[-1] == ")":
                tmp_str += " " + s
                while s[-1] == ")":
                    q_context.pop()
                    s = s[:-1]
                if len(q_context) == 0:
                    tmp_str = tmp_str.strip()
                    current_derivation.append(tmp_str)
                    if len(productions) == 1 and len(current_derivation) == len(
                        self.constraints(productions[0].rhs()[0])
                    ):
                        break
                    tmp_str = ""
            elif len(q_context) > 0:
                tmp_str += " " + s
        current_derivation.append(None)  # type: ignore[arg-type]
        rhs_counter += 1
        q_context = deque()
        if len(productions) == 1 and len(current_derivation) == len(
            self.constraints(productions[0].rhs()[0])
        ):
            pass
        else:
            for s in post_subtree.split(" "):
                if s == "":
                    continue
                elif s[0] == "(":
                    if len(q_context) == 0 and len(s) > 1:
                        productions = [
                            production
                            for production in productions
                            if [
                                str(prod_sym)
                                for prod_sym in production.rhs()
                                if isinstance(prod_sym, Nonterminal)
                            ][rhs_counter]
                            == s[1:]
                        ]
                        rhs_counter += 1
                    q_context.append(s)
                    tmp_str += " " + s
                elif s[-1] == ")":
                    tmp_str += " " + s
                    while s[-1] == ")":
                        if len(q_context) > 0:
                            q_context.pop()
                        s = s[:-1]
                    if len(q_context) == 0:
                        tmp_str = tmp_str.strip()
                        current_derivation.append(tmp_str)
                        if len(productions) == 1 and len(current_derivation) == len(
                            self.constraints(productions[0].rhs()[0])
                        ):
                            break
                        tmp_str = ""
                elif len(q_context) > 0:
                    tmp_str += " " + s

        return topology, current_derivation

    def mutate(
        self, parent: str, subtree_index: int, subtree_node: str, patience: int = 50
    ):
        # chop out subtree
        pre, _, post = self.remove_subtree(parent, subtree_index)
        if pre != " " and bool(post):
            if self.constraint_is_class:
                not_allowed_productions = self.constraints.mutate_not_allowed_productions(
                    subtree_node,
                    pre,
                    post,
                    possible_productions=self.productions(lhs=Nonterminal(subtree_node)),
                )
            else:
                rhs, current_derivation = self._compute_current_context(pre, post)
                context_information = self.constraints(
                    rhs,
                    current_derivation,
                )
                if isinstance(context_information, list):
                    not_allowed_productions = self._get_not_allowed_productions(
                        self.productions(lhs=Nonterminal(subtree_node)),
                        context_information[
                            next(
                                i for i, cd in enumerate(current_derivation) if cd is None
                            )
                        ],
                    )
                elif isinstance(context_information, bool):
                    not_allowed_productions = self._get_not_allowed_productions(
                        self.productions(lhs=Nonterminal(subtree_node)),
                        context_information,
                    )
                else:
                    raise NotImplementedError
        else:
            not_allowed_productions = []
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
            if (
                len(self.productions(lhs=Nonterminal(subtree_node)))
                - len(not_allowed_productions)
                == 1
            ):
                break
            _patience -= 1
        return child.strip()
