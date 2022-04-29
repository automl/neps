import itertools
from collections import deque
from copy import deepcopy
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
        user_priors: bool = False,
    ):
        if start_symbol is None:
            start_symbol = self.start()
        else:
            start_symbol = Nonterminal(start_symbol)

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
        productions = self.productions(lhs=symbol)
        if not_allowed_productions is not None:
            productions = list(
                filter(
                    lambda production: production not in not_allowed_productions,
                    productions,
                )
            )

        if len(productions) == 0:
            raise Exception(f"There is no production possible for {symbol}")

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
                ret_val = self._constrained_sampler(
                    sym, not_allowed_productions, user_priors=user_priors
                )
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

        q_context: Deque = deque()
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
                    tmp_str = self._remove_empty_spaces(tmp_str)
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
                        tmp_str = self._remove_empty_spaces(tmp_str)
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
            _patience -= 1
        child = self._remove_empty_spaces(child)
        return child

    def crossover(
        self,
        parent1: str,
        parent2: str,
        patience: int = 50,
        return_crossover_subtrees: bool = False,
    ):
        _patience = patience
        while _patience > 0:
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
            first_try = True
            while first_try or _patience % 10 != 0:
                first_try = False
                # sample subtree from donor
                donor_subtree_index = self.rand_subtree_fixed_head(parent2, subtree_node)
                # if no subtrees with right head node return False
                if not donor_subtree_index:
                    _patience -= 1
                else:
                    donor_pre, donor_sub, donor_post = self.remove_subtree(
                        parent2, donor_subtree_index
                    )
                    if sub == donor_sub:  # ensure that there is really a crossover
                        _patience -= 1
                        continue
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
                            [i for i, cd in enumerate(current_derivation) if cd is None][
                                0
                            ]
                        ],
                    )
                    if (
                        any(
                            prod.rhs()[0] == "zero"
                            for prod in parent1_not_allowed_productions
                        )
                        and "zero" in donor_sub
                        and donor_sub.count("(") == 1
                        and donor_sub.count(")") == 1
                    ):
                        _patience -= 1
                        continue
                    if (
                        any(
                            prod.rhs()[0] == "zero"
                            for prod in parent2_not_allowed_productions
                        )
                        and "zero" in sub
                        and sub.count("(") == 1
                        and sub.count(")") == 1
                    ):
                        _patience -= 1
                        continue
                    # return the two new tree
                    child1 = pre + donor_sub + post
                    child2 = donor_pre + sub + donor_post
                    child1 = self._remove_empty_spaces(child1)
                    child2 = self._remove_empty_spaces(child2)

                    if return_crossover_subtrees:
                        return (
                            child1,
                            child2,
                            (pre, sub, post),
                            (donor_pre, donor_sub, donor_post),
                        )
                    return child1, child2

        raise Exception("Cannot do crossover")

    @property
    def compute_space_size(self) -> int:
        """Computes the size of the space described by the grammar.

        Args:
            primitive_nonterminal (str, optional): The primitive nonterminal of the grammar. Defaults to "OPS".

        Returns:
            int: size of space described by grammar.
        """

        def recursive_worker(nonterminal: Nonterminal, memory_bank: dict = None) -> int:
            def _get_all_variants(production):
                variants = [production]
                nonterminals = [
                    i
                    for i, sym in enumerate(production.rhs())
                    if isinstance(sym, Nonterminal)
                ]
                max_zero_op = len(nonterminals)
                for n_zero_op in range(1, max_zero_op):
                    for zero_combination in itertools.combinations(
                        nonterminals, n_zero_op
                    ):
                        current_derivation = self.constraints(production.rhs()[0])
                        counter = 0
                        valid_production = True
                        for i, sym in enumerate(production.rhs()):
                            if not isinstance(sym, str):
                                context_information = self.constraints(
                                    production.rhs()[0],
                                    current_derivation,
                                )
                                not_allowed_productions = (
                                    self._get_not_allowed_productions(
                                        self.productions(lhs=sym),
                                        context_information[counter],
                                    )
                                )
                                if (
                                    i in zero_combination
                                    and len(not_allowed_productions) > 0
                                ):
                                    valid_production = False
                                    break
                                if i >= max(zero_combination):
                                    break
                                if i in zero_combination:
                                    current_derivation[counter] = self.none_operation
                                counter += 1
                        if valid_production:
                            new_production = deepcopy(production)
                            rhs = list(new_production.rhs())
                            # pylint: disable=protected-access
                            new_production._rhs = tuple(
                                self.none_operation if i in zero_combination else r
                                for i, r in enumerate(rhs)
                            )
                            # pylint: enable=protected-access
                            variants.append(new_production)
                return variants

            if memory_bank is None:
                memory_bank = {}

            _potential_productions = self.productions(lhs=nonterminal)
            potential_productions = []
            for potential_production in _potential_productions:
                nonterminals = list(
                    {
                        sym
                        for sym in potential_production.rhs()
                        if isinstance(sym, Nonterminal)
                    }
                )
                if any(
                    production.rhs()[0] == self.none_operation
                    for nonterminal in nonterminals
                    for production in self.productions(nonterminal)
                ):
                    potential_productions += _get_all_variants(potential_production)
                elif not (
                    len(potential_production.rhs()) == 1
                    and potential_production.rhs()[0] == self.none_operation
                ):
                    potential_productions.append(potential_production)
            _possibilites = 0
            for potential_production in potential_productions:
                nonterminals = [
                    rhs_sym
                    for rhs_sym in potential_production.rhs()
                    if isinstance(rhs_sym, Nonterminal)
                ]
                possibilities_per_edge = [
                    memory_bank[str(e_nonterminal)]
                    if str(e_nonterminal) in memory_bank.keys()
                    else recursive_worker(e_nonterminal, memory_bank)
                    for e_nonterminal in nonterminals
                ]
                memory_bank.update(
                    {
                        str(e_nonterminal): possibilities_per_edge[i]
                        for i, e_nonterminal in enumerate(nonterminals)
                    }
                )
                product = 1
                for p in possibilities_per_edge:
                    product *= p
                _possibilites += product
            return _possibilites

        return recursive_worker(self.start())
