from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias
from typing_extensions import assert_never

import more_itertools
import networkx as nx
from torch import nn

from neps.exceptions import NePSError

if TYPE_CHECKING:
    import numpy as np


class ParseError(NePSError):
    pass


class ReLUConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return self


class Leaf(NamedTuple):
    symbol: str
    op: Callable


class Container(NamedTuple):
    symbol: str
    children: list[Node]
    op: Callable


class Passthrough(NamedTuple):
    symbol: str
    children: list[Node]


Node: TypeAlias = Container | Passthrough | Leaf


@dataclass
class Tree:
    root: Container | Leaf

    nodes: dict[int, Node]

    children_ids_of: dict[int, list[int]]
    parent_id_of: dict[int, int]
    leafs: list[int]

    @classmethod
    def from_node(cls, node: Node) -> Tree:
        """Create a `Tree` from a node, where node is considered the root."""
        nodes: dict[int, Node] = {}
        children_ids_of: dict[int, list[int]] = {}
        parent_id_of: dict[int, int] = {}

        def _traverse(n: Node, parent_id: int | None = None) -> None:
            node_id = id(n)
            nodes[node_id] = n

            if parent_id is not None:
                parent_id_of[node_id] = parent_id
                children_ids_of[parent_id].append(node_id)

            match n:
                case Leaf():
                    pass
                case Container(_, children, _) | Passthrough(_, children):
                    children_ids_of[node_id] = []
                    for child in children:
                        _traverse(child, node_id)
                case _:
                    assert_never(n)

        _traverse(node)

        # Validate node is a Container or Leaf
        if not isinstance(node, Container | Leaf):
            raise ValueError("Root node must be a Container or Leaf")

        return cls(
            root=node,
            nodes=nodes,
            children_ids_of=children_ids_of,
            parent_id_of=parent_id_of,
            leafs=[nid for nid, n in nodes.items() if isinstance(n, Leaf)],
        )


@dataclass
class Grammar:
    rules: dict[str, Terminal | NonTerminal]

    class Terminal(NamedTuple):
        op: Callable
        shared: bool = False

    class NonTerminal(NamedTuple):
        choices: list[str]
        op: Callable | None = None
        shared: bool = False

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

                    rules[symbol] = Grammar.NonTerminal(choices, None, shared=False)

                case op if callable(op):
                    # > e.g. "S": op
                    rules[symbol] = Grammar.Terminal(op, shared=False)
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
    rng: np.random.Generator,
    variables: dict[str, Node] | None = None,
) -> Node:
    variables = variables or {}
    rule = grammar.rules.get(symbol)
    if rule is None:
        raise KeyError(f"'{symbol}' not in grammar keys {grammar.rules.keys()}")

    shared_node = variables.get(symbol)
    if shared_node is not None:
        return shared_node

    match rule:
        case Grammar.Terminal(op):
            node = Leaf(symbol, op)
        case Grammar.NonTerminal(choices, op):
            chosen_children = rng.choice(choices).split(" ")
            children = [
                sample_grammar(child_symbol, grammar, rng=rng, variables=variables)
                for child_symbol in chosen_children
            ]
            if op is None:
                node = Passthrough(symbol, children=children)
            else:
                node = Container(symbol, op=op, children=children)
        case _:
            assert_never(rule)

    if rule.shared:
        variables[symbol] = node

    return node


def to_node_from_graph(graph: nx.DiGraph, grammar: Grammar) -> Node:
    # Find the unique root (a node with no incoming edges)
    _root = next((n for n, d in graph.in_degree if d == 0), None)
    if _root is None:
        raise ValueError(
            "Could not find a root in the given graph (a node with indegree 0)."
        )

    variables: dict[str, Node] = {}

    def _recurse(node_id: int) -> Node:
        symbol = graph.nodes[node_id].get("label")
        if symbol is None:
            raise ValueError(f"Node {node_id} does not have a 'label' property.")

        shared_node = variables.get(symbol)
        if shared_node is not None:
            return shared_node

        rule = grammar.rules.get(symbol)
        if rule is None:
            raise ValueError(
                f"Symbol '{symbol}' not found in grammar rules: {grammar.rules.keys()}"
            )

        # Based on the type of rule, construct the proper node
        match rule:
            case Grammar.Terminal(op=op):
                node = Leaf(symbol, op)
            case Grammar.NonTerminal(choices=_, op=op):
                children = [_recurse(child_id) for child_id in graph.successors(node_id)]
                if op is None:
                    node = Passthrough(symbol, children)
                else:
                    node = Container(symbol, children, op)
            case _:
                raise ValueError(f"Unexpected rule type for symbol '{symbol}': {rule}")

        if rule.shared:
            variables[symbol] = node

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
    tree: Tree = Tree.from_node(node=root)

    # Note, we can have duplicates here, that's fine, we want to weight those
    # parents with many leafs more heavily... TODO: Maybe?
    parents: list[int] = [tree.parent_id_of[leaf] for leaf in tree.leafs]

    chosen_node_id: int = rng.choice(parents)
    chosen_node: Node = tree.nodes[chosen_node_id]

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


# TODO: This has issues as we are using id's, while we may have heirarchical components
# which share the same id.
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
    match root:
        case Leaf():
            nodes.append((root_id, {"label": root.symbol}))
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


def parse(grammar: Grammar, string: str, *, strict: bool = True) -> Node:
    bracket_stack: list[int] = []
    bracket_pairs: dict[int, int] = {}
    for i, c in enumerate(string):
        match c:
            case "(":
                bracket_stack.append(i)
            case ")":
                if len(bracket_stack) == 0:
                    raise ParseError(
                        f"Encountered mismatched brackets at position {i}"
                        f" in string '{string}'"
                    )
                bracket_start = bracket_stack.pop(-1)
                bracket_pairs[bracket_start] = i
            case _:
                continue

    if len(bracket_stack) > 0:
        raise ParseError(
            "Encountered a mismatch in the number of brackets."
            f"The bracket(s) at position {bracket_stack} were never closed"
            f" in the string '{string}'"
        )

    variables: dict[str, Node] = {}

    def _parse(frm: int, to: int) -> Iterator[Node]:  # noqa: C901, PLR0912, PLR0915
        symbol = ""
        i = frm
        while i <= to:  # Use a while loop as we may jump ahead in the loop
            c = string[i]
            match c:
                # Ignore whiespace
                case " " | "\n" | "\t":
                    i += 1
                # > Ignore, e.g. s(s(a), b) ... In this case, we already parsed
                # out a symbol from the s(a). Should only occur after a ")"
                case "," if symbol == "":
                    assert string[i - 1] == ")"
                    i += 1
                # If the last character of a substring ends in a comma, this
                # is not a valid string.
                case "," if i == to:
                    raise ParseError(
                        "Got a (sub)string terminating in a ','."
                        " The ',' indicates something should come after it."
                        f" {string[frm : to + 1]}"
                    )
                # Otherwise, it's a valid ',' with a symbol before it
                case ",":
                    i += 1
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
                        case Grammar.Terminal(op):
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
                        f" symbol parsed before it in string {string[frm : to + 1]} "
                    )
                # Open a new subtree
                case "(":
                    assert i in bracket_pairs

                    # Find out where we need to parse to get the children
                    bracket_start = i
                    bracket_end = bracket_pairs[bracket_start]
                    assert bracket_end <= to, f"{bracket_end=} > {to=}"
                    children = list(_parse(frm=bracket_start + 1, to=bracket_end))

                    # Advance the tokenizer past the end of that bracket
                    i = bracket_end + 1

                    # Reset the symbol
                    node_symbol = symbol
                    symbol = ""

                    # Build the node with it's children
                    rule = grammar.rules.get(node_symbol)
                    match rule:
                        case Grammar.NonTerminal(_, op):
                            if strict:
                                child_substring = " ".join(
                                    child.symbol for child in children
                                )
                                if child_substring not in rule.choices:
                                    substring = string[bracket_start : bracket_end + 1]
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
                        case Grammar.Terminal(op):
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
                    i += 1
                    continue
                case ")":
                    # If we reached this bracket, just make sure the parsing algorithm
                    # is working correctly by checking we are indeed where we think
                    # we should be which is at `to`
                    assert i == to
                    i += 1

                    node_symbol = symbol
                    symbol = ""  # This should be the end of the recursed call anywho

                    rule = grammar.rules.get(node_symbol)
                    match rule:
                        case Grammar.Terminal(op):
                            node = Leaf(node_symbol, op)
                        case Grammar.NonTerminal(_, op):
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
                    i += 1
                    symbol += c  # Append to current token

        # This occurs when we did not encounter any special characters
        # like `,`, `(` or `)`.
        # I'm pretty sure the only case this can happen is if we have something
        # like the string `"b"`, i.e. just a `Leaf`
        if symbol != "":
            rule = grammar.rules.get(symbol)
            match rule:
                case Grammar.Terminal(op):
                    node = Leaf(symbol, op)
                case Grammar.NonTerminal(_, op):
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

    itr = _parse(frm=0, to=len(string) - 1)
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
    # for example, consider the following, where L1 is shared.
    # L1 -> x -> ... -> L1 -> x -> ...
    already_shared = already_shared or set()
    if rule.shared and node.symbol in already_shared:
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
    match node:
        case Leaf(symbol):
            return symbol
        case Passthrough(symbol, children) | Container(symbol, children):
            return f"{symbol}({', '.join(to_string(c) for c in children)})"
        case _:
            assert_never(node)


def dfs_node(node: Node) -> Iterator[Node]:
    stack: list[Node] = [node]
    while stack:
        nxt = stack.pop(-1)
        yield nxt
        match nxt:
            case Leaf():
                pass
            case Passthrough(_, children) | Container(_, children):
                yield nxt
                stack.extend(reversed(children))


def bfs_node(node: Node) -> Iterator[Node]:
    queue: list[Node] = [node]
    while queue:
        nxt = queue.pop(0)
        yield nxt
        match nxt:
            case Leaf():
                pass
            case Passthrough(_, children) | Container(_, children):
                yield nxt
                queue.extend(children)


# TODO: The variables thing can mess up the max depth
def bfs_grammar(
    grammar: Grammar,
    symbol: str,
    *,
    max_depth: int,
    current_depth: int = 0,
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

    nxt_depth = current_depth + 1

    rule = grammar.rules.get(symbol)
    match rule:
        case Grammar.Terminal(op):
            node = Leaf(symbol, op)
            if rule.shared:
                variables[symbol] = node
            yield node
        case Grammar.NonTerminal(choices, op):
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
                # import rich

                # rich.print(flat_children)
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
    "O": ["3", "1", "id"],
    "reluconvbn": partial(
        ReLUConvBN, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
    ),
    "id": Identity,
    "3": partial(
        nn.Conv2d, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
    ),
    "1": partial(
        nn.Conv2d, in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0
    ),
}


# https://stackoverflow.com/a/29597209
def hierarchy_pos(
    G: nx.DiGraph,
    root: int,
    width: float = 1.0,
    vert_gap: float = 0.2,
    vert_loc: float = 0,
    xcenter: float = 0.5,
) -> dict[int, tuple[float, float]]:
    """From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
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
        width=1.0,
        vert_gap=0.2,
        vert_loc: float = 0,
        xcenter=0.5,
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
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
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
