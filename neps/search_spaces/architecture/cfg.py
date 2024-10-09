from __future__ import annotations

import math
from typing import Hashable

import numpy as np
from nltk import CFG, Production
from nltk.grammar import Nonterminal


class Grammar(CFG):
    """Extended context free grammar (CFG) class from the NLTK python package
    We have provided functionality to sample from the CFG.
    We have included generation capability within the class (before it was an external function)
    Also allow sampling to return whole trees (not just the string of terminals).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # store some extra quantities needed later
        non_unique_nonterminals = [str(prod.lhs()) for prod in self.productions()]
        self.nonterminals = list(set(non_unique_nonterminals))
        self.terminals = list(
            {str(individual) for prod in self.productions() for individual in prod.rhs()}
            - set(self.nonterminals)
        )
        # collect nonterminals that are worth swapping when doing genetic operations (i.e not those with a single production that leads to a terminal)
        self.swappable_nonterminals = list(
            {i for i in non_unique_nonterminals if non_unique_nonterminals.count(i) > 1}
        )

        self._prior = None

        if len(set(self.terminals).intersection(set(self.nonterminals))) > 0:
            raise Exception(
                f"Same terminal and nonterminal symbol: {set(self.terminals).intersection(set(self.nonterminals))}!"
            )
        for nt in self.nonterminals:
            if len(self.productions(Nonterminal(nt))) == 0:
                raise Exception(f"There is no production for nonterminal {nt}")

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

    def sampler(
        self,
        n=1,
        start_symbol: str | None = None,
        user_priors: bool = False,
    ):
        # sample n sequences from the CFG
        # cfactor: the factor to downweight productions (cfactor=1 returns to naive sampling strategy)
        #          smaller cfactor provides smaller sequences (on average)

        # Note that a simple recursive traversal of the grammar (setting convergent=False) where we choose
        # productions at random, often hits Python's max recursion depth as the longer a sequnce gets, the
        # less likely it is to terminate. Therefore, we set the default sampler (setting convergent=True) to
        # downweight frequent productions when traversing the grammar.
        # see https://eli.thegreenplace.net/2010/01/28/generating-random-sentences-from-a-context-free-236grammar
        start_symbol = self.start() if start_symbol is None else Nonterminal(start_symbol)

        return [
            f"{self._sampler(symbol=start_symbol, user_priors=user_priors)})"
            for _ in range(n)
        ]

    def _sampler(
        self,
        symbol=None,
        user_priors: bool = False,
        *,
        _cache: dict[Hashable, str] | None = None,
    ):
        # simple sampler where each production is sampled uniformly from all possible productions
        # Tree choses if return tree or list of terminals
        # recursive implementation
        if _cache is None:
            _cache = {}

        # init the sequence
        tree = "(" + str(symbol)
        # collect possible productions from the starting symbol
        productions = self.productions(lhs=symbol)
        # sample
        if len(productions) == 0:
            raise Exception(f"Nonterminal {symbol} has no productions!")
        if user_priors and self._prior is not None:
            production = choice(productions, probs=self._prior[str(symbol)])
        else:
            production = choice(productions)

        for sym in production.rhs():
            if isinstance(sym, str):
                ## if terminal then add string to sequence
                tree = tree + " " + sym
            else:
                cached = _cache.get(sym)
                if cached is None:
                    cached = self._sampler(sym, user_priors=user_priors, _cache=_cache)
                    _cache[sym] = cached

                tree = tree + " " + cached + ")"

        return tree

    def compute_prior(self, string_tree: str, log: bool = True) -> float:
        prior_prob = 1.0 if not log else 0.0

        symbols = self.nonterminals + self.terminals
        q_production_rules: list[tuple[list, int]] = []
        non_terminal_productions: dict[str, list[Production]] = {
            sym: self.productions(lhs=Nonterminal(sym)) for sym in self.nonterminals
        }

        _symbols_by_size = sorted(symbols, key=len, reverse=True)
        _longest = len(_symbols_by_size[0])

        i = 0
        _tree_len = len(string_tree)
        while i < _tree_len:
            char = string_tree[i]
            if char in " \t\n":
                i += 1
                continue

            if char == "(":
                if i == 0:
                    i += 1
                    continue

                # special case: "(" is (part of) a terminal
                if string_tree[i - 1 : i + 2] != " ( ":
                    i += 1
                    continue

            if char == ")" and string_tree[i - 1] != " ":
                # closing symbol of production
                production = q_production_rules.pop()[0][0]
                lhs_production = production.lhs()

                idx = self.productions(lhs=lhs_production).index(production)
                if log:
                    prior_prob += np.log(self.prior[(lhs_production)][idx] + 1e-15)
                else:
                    prior_prob *= self.prior[str(lhs_production)][idx]
                i += 1
                continue

            _s = string_tree[i : i + _longest]
            for sym in _symbols_by_size:
                if _s.startswith(sym):
                    break
            else:
                raise RuntimeError(
                    f"Terminal or nonterminal at position {i} does not exist"
                )

            i += len(sym) - 1

            if sym in self.terminals:
                _productions, _count = q_production_rules[-1]
                new_productions = [
                    production
                    for production in _productions
                    if production.rhs()[_count] == sym
                ]
                q_production_rules[-1] = (new_productions, _count + 1)
            elif sym in self.nonterminals:
                if len(q_production_rules) > 0:
                    _productions, _count = q_production_rules[-1]
                    new_productions = [
                        production
                        for production in _productions
                        if str(production.rhs()[_count]) == sym
                    ]
                    q_production_rules[-1] = (new_productions, _count + 1)

                q_production_rules.append((non_terminal_productions[sym], 0))
            else:
                raise Exception(f"Unknown symbol {sym}")
            i += 1

        if len(q_production_rules) > 0:
            raise Exception(f"Error in prior computation for {string_tree}")

        return prior_prob

    def mutate(
        self, parent: str, subtree_index: int, subtree_node: str, patience: int = 50
    ) -> str:
        """Grammar-based mutation, i.e., we sample a new subtree from a nonterminal
        node in the parse tree.

        Args:
            parent (str): parent of the mutation.
            subtree_index (int): index pointing to the node that is root of the subtree.
            subtree_node (str): nonterminal symbol of the node.
            patience (int, optional): Number of tries. Defaults to 50.

        Returns:
            str: mutated child from parent.
        """
        # chop out subtree
        pre, _, post = self.remove_subtree(parent, subtree_index)
        _patience = patience
        while _patience > 0:
            # only sample subtree -> avoids full sampling of large parse trees
            new_subtree = self.sampler(1, start_symbol=subtree_node)[0]
            child = pre + new_subtree + post
            if parent != child:  # ensure that parent is really mutated
                break
            _patience -= 1

        return child.strip()

    def crossover(
        self,
        parent1: str,
        parent2: str,
        patience: int = 50,
        return_crossover_subtrees: bool = False,
    ):
        # randomly swap subtrees in two trees
        # if no suitiable subtree exists then return False
        subtree_node, subtree_index = self.rand_subtree(parent1)
        # chop out subtree
        pre, sub, post = self.remove_subtree(parent1, subtree_index)
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
                # return the two new tree
                child1 = pre + donor_sub + post
                child2 = donor_pre + sub + donor_post

                child1 = child1.strip()
                child2 = child2.strip()

                if return_crossover_subtrees:
                    return (
                        child1,
                        child2,
                        (pre, sub, post),
                        (donor_pre, donor_sub, donor_post),
                    )
                return child1, child2

        return False, False

    def rand_subtree(self, tree: str) -> tuple[str, int]:
        """Helper function to choose a random subtree in a given parse tree.
        Runs a single pass through the tree (stored as string) to look for
        the location of swappable nonterminal symbols.

        Args:
            tree (str): parse tree.

        Returns:
            Tuple[str, int]: return the parent node of the subtree and its index.
        """
        split_tree = tree.split(" ")
        swappable_indices = [
            i
            for i in range(len(split_tree))
            if split_tree[i][1:] in self.swappable_nonterminals
        ]
        r = np.random.randint(1, len(swappable_indices))
        chosen_non_terminal = split_tree[swappable_indices[r]][1:]
        chosen_non_terminal_index = swappable_indices[r]
        return chosen_non_terminal, chosen_non_terminal_index

    @staticmethod
    def rand_subtree_fixed_head(
        tree: str, head_node: str, swappable_indices: list | None = None
    ) -> int:
        # helper function to choose a random subtree from a given tree with a specific head node
        # if no such subtree then return False, otherwise return the index of the subtree

        # single pass through tree (stored as string) to look for the location of swappable_non_terminmals
        if swappable_indices is None:
            split_tree = tree.split(" ")
            swappable_indices = [
                i for i in range(len(split_tree)) if split_tree[i][1:] == head_node
            ]
        if not isinstance(swappable_indices, list):
            raise TypeError("Expected list for swappable indices!")
        if len(swappable_indices) == 0:
            # no such subtree
            return False
        else:
            # randomly choose one of these non-terminals
            r = (
                np.random.randint(1, len(swappable_indices))
                if len(swappable_indices) > 1
                else 0
            )
            return swappable_indices[r]

    @staticmethod
    def remove_subtree(tree: str, index: int) -> tuple[str, str, str]:
        """Helper functioon to remove a subtree from a parse tree
        given its index.
        E.g. '(S (S (T 2)) (ADD +) (T 1))'
        becomes '(S (S (T 2)) ', '(T 1))'  after removing (ADD +).

        Args:
            tree (str): parse tree
            index (int): index of the subtree root node

        Returns:
            Tuple[str, str, str]: part before the subtree, subtree, part past subtree
        """
        split_tree = tree.split(" ")
        pre_subtree = " ".join(split_tree[:index]) + " "
        #  get chars to the right of split
        right = " ".join(split_tree[index + 1 :])
        # remove chosen subtree
        # single pass to find the bracket matching the start of the split
        counter, current_index = 1, 0
        for char in right:
            if char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
            if counter == 0:
                break
            current_index += 1
        post_subtree = right[current_index + 1 :]
        removed = "".join(split_tree[index]) + " " + right[: current_index + 1]
        return (pre_subtree, removed, post_subtree)


# helper function for quickly getting a single sample from multinomial with probs
def choice(options, probs=None):
    x = np.random.rand()
    if probs is None:
        # then uniform probs
        num = len(options)
        probs = [1 / num] * num
    cum = 0
    choice = -1
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            choice = i
            break
    return options[choice]
