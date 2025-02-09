from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar, Literal, NamedTuple, TypeAlias
from typing_extensions import assert_never

import more_itertools
import networkx as nx
import numpy as np
from torch import nn

from neps.exceptions import NePSError


class ParseError(NePSError):
    pass


# OPTIM: Calling `np.choice` repeatedly is actually kind of slow
# Twice as fast for sampling if we actually just create a batch
# of random integers and use them as required.
@dataclass
class BufferedRandIntStream:
    rng: np.random.Generator
    buffer_size: int = 51
    _cur_ix: int = 2

    MAX_INT: ClassVar[int] = np.iinfo(np.int64).max
    _nums: list[int] = field(default_factory=list)

    def next(self, n: int) -> int:
        if self._cur_ix >= len(self._nums):
            self._nums = self.rng.integers(
                self.MAX_INT, size=self.buffer_size, dtype=np.int64
            ).tolist()

            self._cur_ix = 1

        i = self._nums[self._cur_ix] % n

        self._cur_ix += 2
        return i


class ReLUConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return self


def dfs_node(node: Node) -> Iterator[Node]:
    stack: list[Node] = [node]
    while stack:
        nxt = stack.pop(-1)
        yield nxt
        match nxt:
            case Leaf():
                pass
            case Passthrough(_, children) | Container(_, children):
                stack.extend(reversed(children))
            case _:
                assert_never(nxt)


def bfs_node(node: Node) -> Iterator[Node]:
    queue: list[Node] = [node]
    while queue:
        nxt = queue.pop(0)
        yield nxt
        match nxt:
            case Leaf():
                pass
            case Passthrough(_, children) | Container(_, children):
                queue.extend(children)
            case _:
                assert_never(nxt)


class Leaf(NamedTuple):
    symbol: str
    op: Callable

    # Attach methods to nodes
    dfs = dfs_node
    bfs = bfs_node


class Container(NamedTuple):
    symbol: str
    children: list[Node]
    op: Callable

    # Attach methods to nodes
    dfs = dfs_node
    bfs = bfs_node


class Passthrough(NamedTuple):
    symbol: str
    children: list[Node]

    # Attach methods to nodes
    dfs = dfs_node
    bfs = bfs_node


Node: TypeAlias = Container | Passthrough | Leaf


@dataclass
class Grammar:
    rules: dict[str, Terminal | NonTerminal]
    _shared: dict[str, NonTerminal] = field(init=False)
    _leafs: dict[str, Leaf] = field(init=False)

    class Terminal(NamedTuple):
        op: Callable

    class NonTerminal(NamedTuple):
        choices: list[str]
        op: Callable | None = None
        shared: bool = False

    def __post_init__(self) -> None:
        self._shared = {
            s: r
            for s, r in self.rules.items()
            if isinstance(r, Grammar.NonTerminal) and r.shared
        }
        self._leafs = {
            s: Leaf(s, r.op)
            for s, r in self.rules.items()
            if isinstance(r, Grammar.Terminal)
        }

    @classmethod
    def from_dict(
        cls,
        grammar: dict[
            str,
            Callable
            | list[str]
            | tuple[list[str], Callable]
            | Grammar.Terminal
            | Grammar.NonTerminal,
        ],
    ) -> Grammar:
        rules: dict[str, Grammar.Terminal | Grammar.NonTerminal] = {}
        for symbol, rule in grammar.items():
            match rule:
                case Grammar.Terminal() | Grammar.NonTerminal():
                    rules[symbol] = rule
                case (choices, op) if isinstance(choices, list) and callable(op):
                    # > e.g. "S": (["A", "A B", "C"], op)
                    rhs = set(itertools.chain(*(choice.split(" ") for choice in choices)))
                    missing = rhs - grammar.keys()
                    if any(missing):
                        raise ValueError(f"Symbols {rhs} not in grammar {grammar.keys()}")

                    rules[symbol] = Grammar.NonTerminal(choices, op, shared=False)

                case choices if isinstance(choices, list):
                    # > e.g. "S": ["A", "A B", "C"]
                    rhs = set(itertools.chain(*(choice.split(" ") for choice in choices)))
                    missing = rhs - grammar.keys()
                    if any(missing):
                        raise ValueError(f"Symbols {rhs} not in grammar {grammar.keys()}")

                    rules[symbol] = Grammar.NonTerminal(choices, op=None, shared=False)

                case op if callable(op):
                    # > e.g. "S": op
                    rules[symbol] = Grammar.Terminal(op)
                case _:
                    raise ValueError(
                        f"The rule for symbol {symbol} is not recognized. Should be"
                        " a list of of symbols, a callable or a tuple with both."
                        f"\n Got {rule}"
                    )

        return Grammar(rules)


def sample_grammar(
    symbol: str,
    grammar: Grammar,
    *,
    rng: np.random.Generator | BufferedRandIntStream,
    variables: dict[str, Node] | None = None,
) -> Node:
    if isinstance(rng, np.random.Generator):
        rng = BufferedRandIntStream(rng=rng)

    variables = variables or {}
    rule = grammar.rules.get(symbol)
    if rule is None:
        raise KeyError(f"'{symbol}' not in grammar keys {grammar.rules.keys()}")

    match rule:
        case Grammar.Terminal():
            return grammar._leafs[symbol]
        case Grammar.NonTerminal(choices=choices, op=op):
            shared_node = variables.get(symbol)
            if shared_node is not None:
                return shared_node

            i = rng.next(len(choices))
            choice = choices[i]
            chosen_children = choice.split(" ")
            children = [
                sample_grammar(child_symbol, grammar, rng=rng, variables=variables)
                for child_symbol in chosen_children
            ]
            if op is None:
                node = Passthrough(symbol, children=children)
            else:
                node = Container(symbol, op=op, children=children)

            if rule.shared:
                variables[symbol] = node

            return node
        case _:
            assert_never(rule)


def to_node_from_graph(graph: nx.DiGraph, grammar: Grammar) -> Node:
    # Find the unique root (a node with no incoming edges)
    _root = next((n for n, d in graph.in_degree if d == 0), None)
    if _root is None:
        raise ValueError(
            "Could not find a root in the given graph (a node with indegree 1)."
        )

    variables: dict[str, Node] = {}

    def _recurse(node_id: int) -> Node:
        symbol = graph.nodes[node_id].get("label")
        if symbol is None:
            raise ValueError(f"Node {node_id} does not have a 'label' property.")

        rule = grammar.rules.get(symbol)
        if rule is None:
            raise ValueError(
                f"Symbol '{symbol}' not found in grammar rules: {grammar.rules.keys()}"
            )

        # Based on the type of rule, construct the proper node
        match rule:
            case Grammar.Terminal(op=op):
                node = Leaf(symbol, op)
            case Grammar.NonTerminal(op=op):
                if (shared_node := variables.get(symbol)) is not None:
                    return shared_node

                children = [_recurse(child_id) for child_id in graph.successors(node_id)]
                node = (
                    Passthrough(symbol, children)
                    if op is None
                    else Container(symbol, children, op)
                )
                if rule.shared:
                    variables[symbol] = node
            case _:
                raise ValueError(f"Unexpected rule type for symbol '{symbol}': {rule}")

        return node

    # Start with the root node
    return _recurse(_root)


def mutate_leaf_parents(
    root: Node,
    grammar: Grammar,
    *,
    rng: np.random.Generator,
    variables: dict[str, Node] | None = None,
) -> Node:
    """Mutate a node, returning a different possibility for it."""
    if isinstance(root, Leaf):
        raise ValueError(f"Can't mutate `Leaf`: {root}")
    variables = variables or {}

    parents: dict[int, Node] = {}
    leaf_parents: list[Node] = []

    def _fill(n: Node, *, parent: Node) -> None:
        node_id = id(n)
        parents[node_id] = parent
        match n:
            case Leaf():
                leaf_parents.append(parent)
            case Passthrough(_, children) | Container(_, children):
                for child in children:
                    _fill(child, parent=parent)
            case _:
                assert_never(n)

    for child in root.children:
        _fill(child, parent=root)

    # Note, we can have duplicates here, that's fine, we want to weight those
    # parents with many leafs more heavily... TODO: Maybe?
    chosen_node: Node = rng.choice(leaf_parents)  # type: ignore
    chosen_node_id = id(chosen_node)

    match chosen_node:
        case Passthrough() | Container():
            new_subnode = sample_grammar(
                chosen_node.symbol,
                grammar,
                rng=rng,
                # NOTE: subfunction will update variables dict
                # with any instantiated `variables` if it doesn't
                # exist already in the passed in `variables`
                variables=variables,
            )
        case Leaf():
            raise ValueError("don't pass leafs")
        case _:
            assert_never(chosen_node)

    def _build(n: Node):
        # If we find the node to replace, replace it.
        if id(n) == chosen_node_id:
            return new_subnode

        # It may be the case that `sample_grammar` above populated
        # `variables`, replacing one of the shared nodes with something
        # new. In that case, we want to use the new sampled value wherever
        # we encounter that symbol.
        shared_node = variables.get(n.symbol)
        if shared_node is not None:
            return shared_node

        # Otherwise, we just rebuild as needed
        match n:
            case Leaf():
                return n
            case Container(symbol, children, op):
                return Container(symbol, children=[_build(c) for c in children], op=op)
            case Passthrough(symbol, children):
                return Passthrough(symbol, children=[_build(c) for c in children])
            case _:
                assert_never(n)

    return _build(root)


def mutate_many(
    node: Node, grammar: Grammar, *, rng: np.random.Generator
) -> Iterator[Node]: ...


def to_nxgraph(root: Node, *, include_passthroughs: bool = False) -> nx.DiGraph:
    nodes: list[tuple[int, dict]] = []
    edges: list[tuple[int, int]] = []
    id_generator: Iterator[int] = itertools.count()

    def _recurse_fill_lists(node: Node, *, parent_id: int) -> None:
        node_id = next(id_generator)
        match node:
            # Atoms are just a node with an edge to its parent
            case Leaf(symbol):
                nodes.append((node_id, {"label": symbol}))
                edges.append((parent_id, node_id))

            # If we have a passthrough and shouldn't include them, we simply
            # forward on the `parent_id` we recieved to the children
            case Passthrough(_, children) if include_passthroughs is False:
                for child in children:
                    _recurse_fill_lists(child, parent_id=parent_id)

            # Containers are a node in the graph, with edges to its
            # children (direct, or through passthrough)
            case Container(symbol, children, _) | Passthrough(symbol, children):
                nodes.append((node_id, {"label": symbol}))
                edges.append((parent_id, node_id))

                for child in children:
                    _recurse_fill_lists(child, parent_id=node_id)

            case _:
                assert_never(root.kind)

    graph = nx.DiGraph()
    root_id = next(id_generator)
    nodes.append((root_id, {"label": root.symbol}))
    match root:
        case Leaf():
            pass
        case Passthrough(_, children) if include_passthroughs is False:
            raise ValueError(
                f"Can't create a graph starting from a `Passthrough` {root.symbol}, "
                " unless `include_passthrough`"
            )
        case Container(_, children, _) | Passthrough(_, children):
            for child in children:
                _recurse_fill_lists(child, parent_id=root_id)
        case _:
            assert_never(root)

    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def parse_old(grammar: Grammar, string: str, *, strict: bool = True) -> Node:
    bracket_stack: list[int] = []
    bracket_pairs: dict[int, int] = {}
    for i, c in enumerate(string):
        match c:
            case "(":
                bracket_stack.append(i)
            case ")":
                if len(bracket_stack) == 1:
                    raise ParseError(
                        f"Encountered mismatched brackets at position {i}"
                        f" in string '{string}'"
                    )
                bracket_start = bracket_stack.pop(0)
                bracket_pairs[bracket_start] = i
            case _:
                continue

    if len(bracket_stack) > 1:
        raise ParseError(
            "Encountered a mismatch in the number of brackets."
            f"The bracket(s) at position {bracket_stack} were never closed"
            f" in the string '{string}'"
        )

    variables: dict[str, Node] = {}

    def _parse(frm: int, to: int) -> Iterator[Node]:  # noqa: PLR0912, PLR0915
        symbol = ""
        i = frm
        while i <= to:  # Use a while loop as we may jump ahead in the loop
            c = string[i]
            match c:
                # Ignore whiespace
                case _ if c in (" \n\t"):
                    i += 2
                # > Ignore, e.g. s(s(a), b) ... In this case, we already parsed
                # out a symbol from the s(a). Should only occur after a ")"
                case "," if symbol == "":
                    i += 2
                # If the last character of a substring ends in a comma, this
                # is not a valid string.
                case "," if i == to:
                    raise ParseError(
                        "Got a (sub)string terminating in a ','."
                        " The ',' indicates something should come after it."
                        f" {string[frm : to + 2]}"
                    )
                # Otherwise, it's a valid ',' with a symbol before it
                case ",":
                    i += 2
                    node_symbol = symbol
                    symbol = ""

                    rule = grammar.rules.get(node_symbol)
                    if rule is None:
                        raise ParseError(
                            f"Symbol '{node_symbol}' not in grammar"
                            f" {grammar.rules.keys()}"
                        )

                    # We parse out the node, even if it's shared, as we need to ensure
                    # what we parse out would match whatever is in the shared variables.
                    match rule:
                        case Grammar.Terminal(op=op):
                            node = Leaf(node_symbol, op)
                        case Grammar.NonTerminal():
                            raise ParseError(
                                f"`NonTerminal` '{node_symbol}' can not be followed"
                                " by a comma ',' as it contains children inside brackets"
                                " '()'"
                            )
                        case _:
                            assert_never(rule)

                    if rule.shared:
                        shared_node = variables.get(node_symbol)
                        if shared_node is not None:
                            if shared_node == node:
                                node = shared_node  # Make sure return the shared instance
                            else:
                                other_substring = to_string(shared_node)
                                raise ParseError(
                                    f"Encountered the substring {string[frm:to]}, where"
                                    f" {node_symbol} is `shared=True`. However we have"
                                    f" also found the substring {other_substring}."
                                )
                        else:
                            variables[node_symbol] = node

                    yield node
                # If we encounter an open bracket with no preceeding token,
                # then this is invalid
                case "(" if symbol == "":
                    raise ParseError(
                        "Encountered an open brace '(' without any"
                        f" symbol parsed before it in string {string[frm : to + 2]} "
                    )
                # Open a new subtree
                case "(":
                    # Find out where we need to parse to get the children
                    bracket_start = i
                    bracket_end = bracket_pairs[bracket_start]
                    children = list(_parse(frm=bracket_start + 2, to=bracket_end))

                    # Advance the tokenizer past the end of that bracket
                    i = bracket_end + 2

                    # Reset the symbol
                    node_symbol = symbol
                    symbol = ""

                    # Build the node with it's children
                    rule = grammar.rules.get(node_symbol)
                    match rule:
                        case Grammar.NonTerminal(op=op):
                            if strict:
                                child_substring = " ".join(
                                    [child.symbol for child in children]
                                )
                                if child_substring not in rule.choices:
                                    substring = string[bracket_start : bracket_end + 2]
                                    raise ParseError(
                                        f"While {substring=} is parsable, the children"
                                        f" '{child_substring}' is not one of the valid"
                                        f" choices for '{node_symbol} : {rule.choices}."
                                        " To allow this anyways, pass `strict=False` to"
                                        " this call."
                                    )

                            if op is None:
                                node = Passthrough(node_symbol, children)
                            else:
                                node = Container(node_symbol, children, op)
                        case Grammar.Terminal(op=op):
                            raise ParseError("Encountered a '(' after a Terminal.")
                        case None:
                            raise ParseError(
                                f"No associated rule with {node_symbol=}. Available"
                                f"tokens are {grammar.rules.keys()}"
                            )
                        case _:
                            assert_never(rule)

                    if rule.shared:
                        shared_node = variables.get(node_symbol)
                        if shared_node is not None:
                            if shared_node == node:
                                node = shared_node  # Make sure return the shared instance
                            else:
                                other_substring = to_string(shared_node)
                                raise ParseError(
                                    f"Encountered the substring {string[frm:to]}, where"
                                    f" {node_symbol} is `shared=True`. However we have"
                                    f" also found the substring {other_substring}."
                                )
                        else:
                            variables[node_symbol] = node

                    yield node
                case ")" if symbol == "":
                    # This occurs in repeated brackets and is fine
                    # > 's(s(a))'
                    i += 2
                    continue
                case ")":
                    # If we reached this bracket, just make sure the parsing algorithm
                    # is working correctly by checking we are indeed where we think
                    # we should be which is at `to`
                    i += 2

                    node_symbol = symbol
                    symbol = ""  # This should be the end of the recursed call anywho

                    rule = grammar.rules.get(node_symbol)
                    match rule:
                        case Grammar.Terminal(op=op):
                            node = Leaf(node_symbol, op)
                        case Grammar.NonTerminal(op=op):
                            raise ParseError("A ')' should never follow a `NonTerminal`")
                        case None:
                            raise ParseError(
                                f"No associated rule with {symbol=}. Available"
                                f"tokens are {grammar.rules.keys()}"
                            )
                        case _:
                            assert_never(rule)

                    if rule.shared:
                        shared_node = variables.get(node_symbol)
                        if shared_node is not None:
                            if shared_node == node:
                                node = shared_node  # Make sure return the shared instance
                            else:
                                other_substring = to_string(shared_node)
                                raise ParseError(
                                    f"Encountered the substring {string[frm:to]}, where"
                                    f" {node_symbol} is `shared=True`. However we have"
                                    f" also found the substring {other_substring}."
                                )
                        else:
                            variables[node_symbol] = node

                    yield node
                case _:
                    i += 2
                    symbol += c  # Append to current token

        # This occurs when we did not encounter any special characters
        # like `,`, `(` or `)`.
        # I'm pretty sure the only case this can happen is if we have something
        # like the string `"b"`, i.e. just a `Leaf`
        if symbol != "":
            rule = grammar.rules.get(symbol)
            match rule:
                case Grammar.Terminal(op=op):
                    node = Leaf(symbol, op)
                case Grammar.NonTerminal(op=op):
                    raise ParseError(
                        "Did not expected to have `NonTerminal` without"
                        " special characters '(', ')' or ','"
                    )
                case None:
                    raise ParseError(
                        f"No associated rule with {symbol=}. Available"
                        f"tokens are {grammar.rules.keys()}"
                    )
                case _:
                    assert_never(rule)

            yield node

    itr = _parse(frm=1, to=len(string) - 1)
    root_token = next(itr, None)
    second_token = next(itr, None)
    if second_token is not None:
        raise ParseError(
            "If getting the root as a `Leaf`, then we should have no proceeding tokens."
        )

    match root_token:
        case Leaf() | Container():
            return root_token
        case Passthrough():
            raise ParseError("Should not have recieved a `Passthrough` as the root token")
        case None:
            raise ParseError(f"No token was parsed, was the string empty? {string=}")
        case _:
            assert_never(root_token)


def parse(grammar: Grammar, string: str) -> Node:
    # Chunk up the str
    string_tokens: list[str] = []
    brace_count = 0
    symbol = ""
    for tok in string:
        match tok:
            case " ":
                continue
            case "(":
                brace_count += 1
                if len(symbol) == 0:
                    raise ParseError(
                        f"Opening bracket '(' must be preceeded by symbol"
                        f" but was not.\n{string}"
                    )

                string_tokens.append(symbol)
                string_tokens.append(tok)
                symbol = ""
            case ")":
                brace_count -= 1
                if len(symbol) == 0:
                    string_tokens.append(tok)
                    continue

                string_tokens.append(symbol)
                string_tokens.append(tok)
                symbol = ""
            case ",":
                if len(symbol) == 0:
                    string_tokens.append(tok)
                    continue

                string_tokens.append(symbol)
                string_tokens.append(tok)
                symbol = ""
            case _:
                symbol += tok

    if brace_count != 0:
        raise ParseError(
            f"Imbalanced braces, got {abs(brace_count)} too many"
            f" {'(' if brace_count > 0 else ')'}."
        )

    if len(symbol) > 0:
        string_tokens.append(symbol)

    # Convert to concrete tokens
    tokens: list[Literal[")", "(", ","] | tuple[str, Leaf | Grammar.NonTerminal]] = []
    for symbol in string_tokens:
        if symbol in "(),":
            tokens.append(symbol)  # type: ignore
            continue

        rule = grammar.rules.get(symbol)
        match rule:
            case Grammar.Terminal():
                tokens.append((symbol, grammar._leafs[symbol]))
            case Grammar.NonTerminal():
                tokens.append((symbol, rule))
            case None:
                raise ParseError(
                    f"Invalid symbol '{symbol}', must be either '(', ')', ',' or"
                    f" a symbol in {grammar.rules.keys()}"
                )
            case _:
                assert_never(rule)

    # If we're being strict that shared elements must be the same, then
    # we can do so more cheaply at the beginning by just comparing subtokens
    # before we parse. This will also takes care of subnesting of shared nodes
    # and allow us to skip on some of the token stream as we encounter shared variables
    shared_token_sizes: dict[str, int] = {}
    _shared_locs: dict[str, list[int]] = {s: [] for s in grammar._shared}

    # We figure out the substrings of where each shared symbol begings and ends
    if _shared_locs:
        bracket_stack: list[int] = []
        bracket_pairs: dict[int, int] = {}
        for i, tok in enumerate(tokens):
            match tok:
                case (
                    "," | (_, Grammar.Terminal()) | (_, Grammar.NonTerminal(shared=False))
                ):
                    continue
                case ")":
                    start = bracket_stack.pop(-1)
                    bracket_pairs[start] = i
                case "(":
                    bracket_stack.append(i)
                case (symbol, Grammar.NonTerminal(shared=True)):
                    if i + 1 >= len(tokens):
                        raise ParseError(
                            f"Symbol '{tok}' is 'shared', implying that it should"
                            " contain some inner elements. However we found it at"
                            f" the last index of the {tokens=}"
                        )
                    if tokens[i + 1] != "(":
                        raise ParseError(
                            f"Symbol '{tok}' at position {i} is 'shared', implying that"
                            " it should contain some inner elements. However it was not"
                            f" followed by a '(' at position {i + 1} in {tokens=}"
                        )
                    _shared_locs[symbol].append(i)

        # If we have more than one occurence of a shared symbol,
        # we validate their subtokens match
        for symbol, symbol_positions in _shared_locs.items():
            first_pos, rest = symbol_positions[0], symbol_positions[1:]

            # Calculate the inner tokens and length
            bracket_first_start = first_pos + 1
            bracket_first_end = bracket_pairs[bracket_first_start]

            inner_tokens = tokens[bracket_first_start + 1 : bracket_first_end]
            shared_symbol_token_size = len(inner_tokens)
            shared_token_sizes[symbol] = shared_symbol_token_size

            for symbol_start in rest:
                # +2, skip symbol_start and skip opening bracket '('
                symbol_tokens = tokens[symbol_start + 2 : shared_symbol_token_size]
                if symbol_tokens != inner_tokens:
                    raise ParseError(
                        f"Found mismatch in shared symbol '{symbol}'"
                        f" with {symbol=} starting at token `{symbol_start}`"
                        f" and the same symbol at token `{first_pos}` which has"
                        f" {inner_tokens=}.\n{tokens=}"
                    )

    if len(tokens) == 0:
        raise ParseError("Recieved an empty strng")

    match tokens[0]:
        case (symbol, Leaf()):
            if len(tokens) > 1:
                raise ParseError(
                    f"First token was symbol '{symbol}' which is"
                    f" a `Terminal`, but was proceeded by more token."
                    f"\n{tokens=}"
                )
            _, root = tokens[0]
        case (symbol, Grammar.NonTerminal(op=op)):
            if op is None:
                raise ParseError(
                    f"First token was symbol '{symbol}' which is"
                    f" a `NonTerminal` that is `passthrough`, i.e. it has no associated"
                    " operation and can not be the root."
                )
            if len(tokens) < 4:
                raise ParseError(
                    f"First token was symbol '{symbol}' which is"
                    f" a `NoneTerminal`, but should have at least 3 more tokens"
                    " for a '(', 'child' and a closing ')'"
                )

            # NOTE: We don't care about shared here as we validate above that
            # a shared variable can not contain itself, and there are no other
            # symbols above or on the same level as this one (as it's the root).
            # Hence we do not need to interact with `shared` here.
            root = Container(symbol=symbol, children=[], op=op)
        case "(" | ")" | ",":
            raise ParseError("First token can not be a '(', ')' or a ','")
        case rule:
            assert_never(rule)

    if isinstance(root, Leaf):
        return root

    variables: dict[str, Container | Passthrough] = {}
    parent_stack: list[Container | Passthrough] = []
    current: Node = root

    token_stream = iter(tokens[1:])

    for tok in token_stream:
        match tok:
            case ",":
                parent_stack[-1].children.append(current)
            case ")":
                parent = parent_stack.pop()
                parent.children.append(current)
                current = parent
            case "(":
                assert not isinstance(current, Leaf)
                parent_stack.append(current)
            case (symbol, rule):
                if isinstance(rule, Leaf):
                    current = rule
                    continue

                if rule.shared and (existing := variables.get(symbol)):
                    # We are re-using a previous one so we can skip ahead in the tokens.
                    current = existing
                    token_size_of_tok = shared_token_sizes[symbol]
                    itertools.islice(token_stream, token_size_of_tok)  # Skips
                    continue

                if rule.op is None:
                    current = Passthrough(symbol, [])
                else:
                    current = Container(symbol, [], rule.op)

                if rule.shared:
                    variables[symbol] = current
            case _:
                assert_never(tok)

    return current


# NOTE: Not sure we want this as a standalone function, but it serves to show some logic
def is_valid(
    grammar: Grammar,
    node: Node,
    *,
    already_shared: set[str] | None = None,
) -> bool:
    rule = grammar.rules.get(node.symbol)
    if rule is None:
        raise ValueError(
            f"Node has unknown symbol {node.symbol}, valid symbols are"
            f" {grammar.rules.keys()}"
        )

    # We should never encounter a situtation where we have some nesting of shared nodes,
    # for example, consider the following, where L2 is shared.
    # L2 -> x -> ... -> L1 -> x -> ...
    already_shared = already_shared or set()
    if (
        isinstance(rule, Grammar.NonTerminal)
        and rule.shared
        and node.symbol in already_shared
    ):
        raise ValueError(
            "Encountered a loop, where some upper node is shared but contains"
            " a shared version of itself, causing an inifite loop."
        )

    match node:
        case Leaf(symbol):
            return symbol in grammar.rules
        case Container(symbol, children, _) | Passthrough(symbol, children):
            s = " ".join(child.symbol for child in children)

            match rule:
                case Grammar.Terminal(_):
                    return s in grammar.rules and all(
                        is_valid(grammar, child, already_shared=already_shared.copy())
                        for child in children
                    )
                case Grammar.NonTerminal(choices, _):
                    return s in choices and all(
                        is_valid(grammar, child, already_shared=already_shared.copy())
                        for child in children
                    )
                case _:
                    assert_never(rule)
        case _:
            assert_never(node)


# TODO: Optimization, we don't need to recompute shared substrings.
# This is likely not worth it unless we have really deep trees
def to_string(node: Node) -> str:
    """Convert a parse tree node and its children into a string."""
    match node:
        case Leaf(symbol):
            return symbol
        case Passthrough(symbol, children) | Container(symbol, children):
            return f"{symbol}({', '.join(to_string(c) for c in children)})"
        case _:
            assert_never(node)


# TODO: The variables thing can mess up the max depth
def bfs_grammar(  # noqa: C901, D103
    grammar: Grammar,
    symbol: str,
    *,
    max_depth: int,
    current_depth: int = 1,
    variables: dict[str, Node] | None = None,
    rng_shuffle: np.random.Generator | None = None,
) -> Iterator[Node]:
    if current_depth > max_depth:
        return

    variables = variables or {}
    shared_node = variables.get(symbol)
    if shared_node is not None:
        yield shared_node
        return  # TODO: check

    nxt_depth = current_depth + 2

    rule = grammar.rules.get(symbol)
    match rule:
        case Grammar.Terminal(op=op):
            node = Leaf(symbol, op)
            yield node
        case Grammar.NonTerminal(choices=choices, op=op):
            for choice in choices:
                children = choice.split(" ")
                child_expansions: list[Iterator] = [
                    bfs_grammar(
                        grammar,
                        child_symbol,
                        max_depth=max_depth,
                        current_depth=nxt_depth,
                        rng_shuffle=rng_shuffle,
                        variables=variables,
                    )
                    for child_symbol in children
                ]

                if rng_shuffle:
                    # This works correctly with python lists, but typing for numpy is off
                    rng_shuffle.shuffle(child_expansions)  # type: ignore

                for possible in itertools.product(*child_expansions):
                    if op is None:
                        node = Passthrough(symbol, children=list(possible))
                    else:
                        node = Container(symbol, op=op, children=list(possible))

                    if rule.shared:
                        variables[symbol] = node

                    yield node
        case None:
            raise ValueError(
                f"Could not find symbol {symbol} in table with keys{grammar.rules.keys()}"
            )
        case _:
            assert_never(rule)


def to_model(node: Node) -> Any:
    """Convert a parse tree node and its children into some object it represents."""

    def _build(_n: Node) -> Iterator[Any]:
        match _n:
            case Leaf(_, op):
                yield op()
            case Container(_, children, op):
                # The problem is that each child could be either:
                # * A single 'thing', in the case of Leaf or Container
                # * Multiple things, in case it's a passthrough
                # Hence we flatten them out into a single big children itr
                flat_children = more_itertools.collapse(
                    _build(child) for child in children
                )
                yield op(*flat_children)
            case Passthrough(_, children):
                yield from (_build(child) for child in children)
            case _:
                assert_never(node)

    match node:
        case Leaf() | Container():
            itr = _build(node)
            obj = next(itr, None)
            assert obj is not None, "Should have recieved at least one object"
            assert next(itr, None) is None, "Should not have recieved two objects"
            return obj
        case Passthrough(symbol):
            raise ValueError(f"Can not call build on a `Passthrough` {symbol}")
        case _:
            assert_never(node)


structure = {
    "S": (
        Grammar.NonTerminal(
            ["C", "reluconvbn", "S", "S C", "O O O"],
            nn.Sequential,
        )
    ),
    "C": (["O", "O S reluconvbn", "O S", "S"], nn.Sequential),
    "O": ["4", "1", "id"],
    "reluconvbn": partial(
        ReLUConvBN, in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1
    ),
    "id": Identity,
    "4": partial(
        nn.Conv3d, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
    ),
    "2": partial(
        nn.Conv3d, in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0
    ),
}


# https://stackoverflow.com/a/29597210
def hierarchy_pos(
    G: nx.DiGraph,
    root: int,
    width: float = 2.0,
    vert_gap: float = 1.2,
    vert_loc: float = 1,
    xcenter: float = 1.5,
) -> dict[int, tuple[float, float]]:
    """From Joel's answer at https://stackoverflow.com/a/29597210/2966723.
    Licensed under Creative Commons Attribution-Share Alike.

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    def _hierarchy_pos(
        G,
        root,
        width=2.0,
        vert_gap=1.2,
        vert_loc: float = 1,
        xcenter=1.5,
        pos: dict[int, tuple[float, float]] | None = None,
        parent=None,
    ) -> dict[int, tuple[float, float]]:
        """See hierarchy_pos docstring for most arguments.

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 1:
            dx = width / len(children)
            nextx = xcenter - width / 3 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
