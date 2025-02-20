"""A module containing the [`Grammar`][neps.space.grammar.Grammar] parameter.

A `Grammar` contains a list of production `rules`, which produce a _string_ from
the grammar, as well as some `start_symbol` which is used by optimizers.

!!! note

    We make a distinction that **string** is not a python `str`, and represents
    an expanded set of rules from the grammar.

Each rule, either a [`Terminal`][neps.space.grammar.Grammar.Terminal] or
[`NonTerminal`][neps.space.grammar.Grammar.NonTerminal], is a key-value pair,
where the key is a symbol, such as `"S"` and the value is what the symbol represents.
See the example below.

You can create a `Grammar` conveninetly using
[`Grammar.from_dict({...})`][neps.space.grammar.Grammar.from_dict].

!!! example

    ```python
    from neps import Grammar

    # Using bare types
    grammar = Grammar.from_dict({
        "S": (["OP OP OP", "OP OP"], nn.Sequential),  # A seq with either 3 or 2 children
        "OP": ["linear golu", "linear relu"], # A choice between linear with a golu/relu
        "linear": partial(nn.LazyLinear, out_features=10, bias=False), # A linear layer
        "relu": nn.ReLU, # A relu activation
        "golu": nn.GoLU, # A golu activation
    })

    # Explicitly
    grammar = Grammar({
        "S": NonTerminal(choices=["OP OP OP"], op=nn.Sequential, shared=False),
        "OP": NonTerminal(choices=["linear golu", "linear relu"], op=None, shared=False),
        "relu": Terminal(nn.ReLU),
        "linear": Terminal(partial(nn.LazyLinear, out_features=10, bias=False)),
        "golu": Terminal(nn.GoLU),
    })
    ```

A _string_ from a `Grammar` can be produced in several ways:

* [`grammar.parse()`][neps.space.grammar.Grammar.parse] - parse a grammar from a `str`
    into a _string_, which is represented by a [`Node`][neps.space.grammar.Node] tree.
    The inverse of this operation is to call `node.to_string()`.
* [`grammar.sample()`][neps.space.grammar.Grammar.sample] - Sample a random string from
    the grammar.
* [`grammar.mutations()`][neps.space.grammar.Grammar.mutations] - This takes in a `Node`,
    which represents a _string_ from the grammar, and can mutate selected points of the
    string. You can use the function [`node.select()`][neps.space.grammar.select] for
    different strategies to select parts of the string to mutate, for example, all
    parents of a leaf with `node.select(how=("climb", 1))` or specific symbols using
    `node.select(how=("symbol", "OP"))`.
* [`grammar.bfs()`][neps.space.grammar.Grammar.bfs] - This iterates through all possible
    strings producable from the grammar, using a max-depth to prevent infinite recursion.

As mentioned in the above methods, a string from the the `Grammar` is represnted as a tree
of [`Node`][neps.space.grammar.Node], which also contain the associated meaning of the
string parts, i.e. what operation that symbol should do.

* [`Leaf`][neps.space.grammar.Leaf] - A symbol with no children and an operation.
* [`Container`][neps.space.grammar.Container] - A symbol with children and some containing
    operation, for example an `nn.Sequential`.
* [`Passthrough`][neps.space.grammar.Passthrough] - A symbol with children but **no**
    operation. It's children will be passed up to its parent until it hits a `Container`.

Please see the associated docstrings for more information.

For the most part, you can consider all of these as a [`Node`][neps.space.grammar.Node],
which has the following attached functions:

* [`to_string()`][neps.space.grammar.to_string] - Convert to it's python `str`
    representation.
* [`to_model()`][neps.space.grammar.to_model] - Convert it into some kind of model,
    defined by its operations. Normally this represnts some `nn.Module` structure but it
    is not necessarily torch specific.
* [`to_nxgraph()`][neps.space.grammar.to_nxgraph] - Convert it into a `nx.Digraph` which
    can be useful for optimization or other applications such as plotting. The inverse
    operation is called from the grammar,
    [`grammar.node_from_nxgraph()`][neps.space.grammar.Grammar.node_from_nxgraph]
* [`select()`][neps.space.grammar.select] - Select certain nodes of the string by
    a criterion.
* [`dfs()`][neps.space.grammar.dfs] - DFS iteration over the nodes of the string.
* [`bfs()`][neps.space.grammar.bfs] - BFS iteration over the nodes of the string.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, NamedTuple, TypeAlias
from typing_extensions import assert_never

import more_itertools
import networkx as nx
import numpy as np

from neps.exceptions import NePSError


class ParseError(NePSError):
    """An error occured while parsing a grammar string."""


@dataclass
class _BufferedRandInts:
    rng: np.random.Generator
    buffer_size: int = 50
    _cur_ix: int = 0

    MAX_INT: ClassVar[int] = np.iinfo(np.int64).max
    _nums: list[int] = field(default_factory=list)

    def next(self, n: int) -> int:
        if self._cur_ix >= len(self._nums):
            self._nums = self.rng.integers(
                self.MAX_INT, size=self.buffer_size, dtype=np.int64
            ).tolist()

            self._cur_ix = 0

        i = self._nums[self._cur_ix] % n

        self._cur_ix += 1
        return i


def dfs_node(node: Node) -> Iterator[Node]:
    """Perform a depth-first search iteration on the node."""
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
    """Perform a breadth-first search iteration on the node."""
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


def to_nxgraph(root: Node, *, include_passthroughs: bool = False) -> nx.DiGraph:  # noqa: C901
    """Convert a node and it's children into an `nx.DiGraph`.

    Args:
        root: The node to start from.
        include_passthroughs: Whether to include passthrough symbols into the
            produced graph.
    """
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


def to_model(node: Node) -> Any:
    """Convert a parse tree node and its children into some object it represents."""

    def _build(_n: Node) -> list[Any] | Any:
        match _n:
            case Leaf(_, op):
                return op()
            case Container(_, children, op):
                # The problem is that each child could be either:
                # * A single 'thing', in the case of Leaf or Container
                # * Multiple things, in case it's a passthrough
                # Hence we flatten them out into a single big children itr
                built_children = more_itertools.collapse(
                    (_build(child) for child in children),
                    base_type=(op if isinstance(op, type) else None),
                )
                return op(*built_children)
            case Passthrough(_, children):
                return [_build(child) for child in children]
            case _:
                assert_never(node)

    match node:
        case Leaf() | Container():
            obj = _build(node)
            assert not isinstance(obj, list)
            return obj
        case Passthrough(symbol):
            raise ValueError(f"Can not call build on a `Passthrough` {symbol}")
        case _:
            assert_never(node)


def select(  # noqa: C901, PLR0912, PLR0915
    root: Node,
    *,
    how: (
        tuple[Literal["symbol"], str]
        | tuple[Literal["depth"], int | range]
        | tuple[Literal["climb"], int | range]
    ),
) -> Iterator[Node]:
    """Iterate through the tree and select nodes according to `how=`.

    Args:
        root: the root node to start from.
        how: which nodes to select. In the case of `"depth"` and `"climb"`, you can either
            provide a specific value `int`, or else a `range`, where anything that has
            a value in that `range` is included. Note that this follows the same
            convention that `4 in range(3, 5)` but `5 not in range(3, 5)`,
            i.e. that the stop boundary is non-inclusive.

            * `"symbol"` - Select all nodes which have the given symbol.
            * `"depth"`- Select all nodes which are at a given depth, either a particular
                depth value or a range of depth values. The `root` is defined to be at
                `depth == 0` while its direct children are defined to be at `depth == 1`.
            * `"climb"`- Select all nodes which are at a given distance away from a leaf.
                Leafs are defined to be at `climb == 0`, while any direct parents
                of a leaf are `climb == 1`.
    """
    match how:
        case ("symbol", symbol):
            for node in bfs_node(root):
                if node.symbol == symbol:
                    yield node
        case ("depth", depth):
            if isinstance(depth, int):
                depth = range(depth, depth + 1)

            queue_depth: list[tuple[Node, int]] = [(root, 0)]
            while queue_depth:
                nxt, d = queue_depth.pop(0)
                if d in depth:
                    yield nxt

                if d >= depth.stop:
                    continue

                match nxt:
                    case Leaf():
                        pass
                    case Passthrough(children=children) | Container(children=children):
                        queue_depth.extend([(child, d + 1) for child in children])
                    case _:
                        assert_never(nxt)

        case ("climb", climb):
            if isinstance(climb, int):
                climb = range(climb, climb + 1)

            # First, we iterate downwards, populating parent paths back
            # up. As the id for a Leaf is shared across all similar leafs
            # as well as the fact shared nodes will share the same node id,
            # we could have multiple parents per child id.
            parents: defaultdict[int, list[Node]] = defaultdict(list)

            # We remove duplicates using a dict and the shared ids, a list would
            # end up with duplicates for every leaf. We use this later to begin
            # the climb iteration
            leafs: dict[int, Node] = {}

            queue_climb: list[Node] = [root]
            while queue_climb:
                nxt = queue_climb.pop(0)
                this_id = id(nxt)
                match nxt:
                    case Leaf():
                        leafs[this_id] = nxt
                    case Passthrough(children=children) | Container(children=children):
                        for child in children:
                            parents[id(child)].append(nxt)
                        queue_climb.extend(children)
                    case _:
                        assert_never(nxt)

            # Now we work backwards from the leafs for each of the possible parents
            # for the node id, yielding if we're within the climb path. If we've gone
            # pass the climb value, we can stop iterating there.
            climb_queue: list[tuple[Node, int]] = []
            climb_queue.extend([(leaf, 0) for leaf in leafs.values()])
            seen: set[int] = set()
            while climb_queue:
                node, climb_value = climb_queue.pop(0)
                node_id = id(node)
                if node_id in seen:
                    continue

                if climb_value in climb:
                    seen.add(node_id)
                    yield node

                if climb_value < climb.stop:
                    possible_node_parents = parents[id(node)]
                    climb_queue.extend(
                        [
                            (p, climb_value + 1)
                            for p in possible_node_parents
                            if id(p) not in seen
                        ]
                    )

        case _:
            assert_never(how)


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
            return None


class Leaf(NamedTuple):
    """A node which has no children.

    !!! note

        As we only ever have one kind of leaf per symbol, we tend to re-use the
        same instance of a `Leaf` which gets re-used where it needs to. In contrast,
        a `Container` and `Passthrough` may have different children per symbol and a new
        instance is made each time.

    Args:
        symbol: The string symbol associated with this `Leaf`.
        op: The associated operations with this `symbol`.
    """

    symbol: str
    op: Callable

    def __hash__(self) -> int:
        return hash(self.symbol)

    dfs = dfs_node
    bfs = bfs_node
    to_string = to_string
    to_nxgraph = to_nxgraph
    to_model = to_model
    select = select


class Container(NamedTuple):
    """A node which contains children and has an associated operation.

    Args:
        symobl: The string symbol associated with this `Container`.
        children: The direct children of this node. When instantiating this container,
            it will be called with it's instantiated children with `op(*children)`.
        op: The associated operation with this node, such as an `nn.Sequential`.
    """

    symbol: str
    children: list[Node]
    op: Callable

    def __hash__(self) -> int:
        return hash(self.symbol) + hash(tuple(self.children))

    dfs = dfs_node
    bfs = bfs_node
    to_string = to_string
    to_nxgraph = to_nxgraph
    to_model = to_model
    select = select


class Passthrough(NamedTuple):
    """A node which contains children but has no associated operation.

    This is used for things such as `"OP": ["conv2d", "conv3d", "identity"]`, where
    `"OP"` does not have some kind of container operation and is used to make a choice
    between various symbols.

    Args:
        symbol: The associated symbol with this `Passthrough`.
        children: The direct children of this node. As this node can not be instantiated,
            the children of this `Passthrough` are forward on to this nodes parents.
    """

    symbol: str
    children: list[Node]

    def __hash__(self) -> int:
        return hash(self.symbol) + hash(tuple(self.children))

    dfs = dfs_node
    bfs = bfs_node
    to_string = to_string
    to_nxgraph = to_nxgraph
    to_model = to_model
    select = select


Node: TypeAlias = Container | Passthrough | Leaf
"""The possible nodes in a constructed instance of a string from the grammar.

Please see the associated types for their description or the docstring of a
[`Grammar`][neps.space.grammar.Grammar].
"""


@dataclass
class Grammar:
    """A grammar defines a search space of symbols which may contain other symbols.

    !!! tip

        You most likely want to create one of these using
        [`from_dict()`][neps.space.grammar.Grammar.from_dict].

    A grammar consists of `rules: dict[str, Grammar.Terminal | Grammar.NonTerminal]`
    where the key is a string symbol, and the values are what that string symbol
    represents. The initial symbol used by optimizers is specified using `start_symbol`.

    The [`Grammar.Terminal`][neps.space.Grammar.Terminal] represents some kind of leaf
    node of a computation graph, such as a function call or some operation which
    does not have any children dependancies, for example an `nn.Linear`. This is
    modeled as a [`Node`][neps.space.grammar.Node], specifically the
    [`Leaf`][neps.space.grammar.Leaf] type.

    The [`Grammar.NonTerminal`][neps.space.Grammar.NonTerminal] represents some kind of
    intermediate operation, which contains sub-symbols which are sub-computations of
    a  computation graph. A common example of this is when `op=nn.Sequential`, which by
    itself does not really do any computations but relies on the computation of it's
    children which it performs one after another. If there is an associated `op=`, then
    we consider this be a [`Container`][neps.space.grammar.Container] kind of
    [`Node`][neps.space.grammar.Node]. If there is **no** associated `op=`, then we
    consider this to be a [`Passthrough`][neps.space.grammar.Passthrough] kind of
    [`Node`][neps.space.grammar.Node].

    For a `Grammar.NonTerminal`, you may also specify if it is `shared: bool`, which is
    by default `False`. When explicitly set as `True`, all choices made for its children
    will be shared through the generated/sampled/parsed string. In constrast, if
    `shared=False`, then any specific instance of this symbol may have different children.

    Args:
        start_symbol: The starting symbol used by optimizers.
        rules: The possible grammar rules which define the structure of the grammar.
    """

    start_symbol: str
    rules: dict[str, Terminal | NonTerminal]
    _shared: dict[str, NonTerminal] = field(init=False)
    _leafs: dict[str, Leaf] = field(init=False)

    class Terminal(NamedTuple):
        """A symbol which has no children and an associated operation.

        When a specific instance of a string from this grammar is made, this
        rule will create a [`Leaf`][neps.space.grammar.Leaf].

        Args:
            op: The associated operation.
        """

        op: Callable

    class NonTerminal(NamedTuple):
        """A symbol which has different possible children.

        Depending on whether `op=` is specified or not, this will either be a
        [`Container`][neps.space.grammar.Container] or a
        [`Passthrough`][neps.space.grammar.Passthrough].

        Args:
            choices: The list of possible children to place inside this `NonTerminal`.
                Different possibilities are specified by the elements of the list.
                When a `str` contains multiple symbols that are space seperated, these
                will both be children.

                ```
                # The following says that we have a choice between "a", "b" and "c d".
                # In the case that "c d" is chosen, both of those will be children of the
                # created node.
                ["a", "b", "c d"]
                ```

            op: The associated operation with this node, if any.
            shared: Whether the choices made for this symbol should be shared throughout
                the tree, or whether they should be considred independant.
        """

        choices: list[str]
        op: Callable | None = None
        shared: bool = False

    def __post_init__(self) -> None:
        start_rule = self.rules.get(self.start_symbol, None)
        if start_rule is None:
            raise ValueError(
                f"The start_symbol '{self.start_symbol}' should be one of the symbols"
                f" in rules, which are {self.rules.keys()}"
            )
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
        start_symbol: str,
        grammar: dict[
            str,
            Callable
            | list[str]
            | tuple[list[str], Callable]
            | Grammar.Terminal
            | Grammar.NonTerminal,
        ],
    ) -> Grammar:
        """Create a `Grammar` from a dictionary.

        Please see the module doc for more.

        Args:
            start_symbol: The starting symbol from which to produce strings.
            grammar: The rules of the grammar.
        """
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

        return Grammar(start_symbol=start_symbol, rules=rules)

    def sample(  # noqa: C901, PLR0912
        self,
        symbol: str | None = None,
        *,
        rng: np.random.Generator | _BufferedRandInts,
        variables: dict[str, Node] | None = None,
    ) -> Node:
        """Sample a random string from this grammar.

        Args:
            symbol: The symbol to start from. If not provided, this will use
                the `start_symbol`.
            rng: The random generator by which sampling is done.
            variables: Any shared variables to use in the case that a sampled
                rule has `shared=True`.

        Returns:
            The root of the sampled string.
        """
        if isinstance(rng, np.random.Generator):
            rng = _BufferedRandInts(rng=rng)

        if symbol is None:
            symbol = self.start_symbol

        variables = variables or {}
        rule = self.rules.get(symbol)
        if rule is None:
            raise KeyError(f"'{symbol}' not in grammar keys {self.rules.keys()}")

        stack: list[Container | Passthrough] = []
        match rule:
            case Grammar.Terminal():
                return self._leafs[symbol]
            case Grammar.NonTerminal(choices, op, shared):
                shared_node = variables.get(symbol)
                if shared_node is not None:
                    return shared_node

                i = rng.next(len(rule.choices))
                initial_sample = rule.choices[i]
                children_symbols = initial_sample.split(" ")
                root = (
                    Passthrough(symbol, []) if op is None else Container(symbol, [], op)
                )
                stack.append(root)
            case _:
                assert_never(rule)

        while stack:
            parent = stack.pop()
            i = rng.next(len(choices))
            choice = choices[i]
            children_symbols = choice.split(" ")

            for child_symbol in children_symbols:
                rule = self.rules[child_symbol]
                match rule:
                    case Grammar.Terminal():
                        parent.children.append(self._leafs[child_symbol])
                    case Grammar.NonTerminal(choices, op, shared):
                        shared_node = variables.get(child_symbol)
                        if shared_node is not None:
                            parent.children.append(shared_node)
                            continue

                        sub_parent = (
                            Passthrough(child_symbol, [])
                            if op is None
                            else Container(child_symbol, [], op)
                        )
                        parent.children.append(sub_parent)
                        stack.append(sub_parent)

                        if shared:
                            variables[child_symbol] = sub_parent
                    case _:
                        assert_never(rule)

        return root

    def node_from_graph(self, graph: nx.DiGraph) -> Node:
        """Convert an `nx.DiGraph` into a string.

        Args:
            graph: The graph, produced by
                [`to_nxgraph()`][neps.space.grammar.Grammar.to_nxgraph]

        Returns:
            The root of the string produced from the graph.
        """
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

            rule = self.rules.get(symbol)
            if rule is None:
                raise ValueError(
                    f"Symbol '{symbol}' not found in grammar rules: {self.rules.keys()}"
                )

            # Based on the type of rule, construct the proper node
            match rule:
                case Grammar.Terminal(op=op):
                    node = Leaf(symbol, op)
                case Grammar.NonTerminal(op=op):
                    if (shared_node := variables.get(symbol)) is not None:
                        return shared_node

                    children = [
                        _recurse(child_id) for child_id in graph.successors(node_id)
                    ]
                    node = (
                        Passthrough(symbol, children)
                        if op is None
                        else Container(symbol, children, op)
                    )
                    if rule.shared:
                        variables[symbol] = node
                case _:
                    raise ValueError(
                        f"Unexpected rule type for symbol '{symbol}': {rule}"
                    )

            return node

        # Start with the root node
        return _recurse(_root)

    def mutations(
        self,
        root: Node,
        *,
        which: Iterable[Node],
        max_mutation_depth: int,
        rng_shuffle: np.random.Generator | None = None,
        variables: dict[str, Node] | None = None,
    ) -> Iterator[Node]:
        """Mutate nodes, returning all the different possibilities for them.

        Args:
            root: The root from which to operate.
            which: What nodes to mutate, look at `select()`.
            max_mutation_depth: The maximum depth allowed for bfs iteration
                on the mutant nodes.
            rng_shuffle: Whether to shuffle the return order. This takes place at the
                place when considering the possibilities for a given node, and does
                not follow the order of `NonTerminal.choices`.
            variables: Any predefined values you'd like for different symbols.

        Returns:
            A new tree per possible mutation
        """
        if isinstance(root, Leaf):
            raise ValueError(f"Can't mutate `Leaf`: {root}")

        variables = variables or {}
        mutation_ids = {id(n) for n in which}

        def _inner(node: Node) -> Iterator[Node]:
            match node:
                case Leaf():
                    # We can't mutate leafs as they don't have possible choices to
                    # choose from # by definition so we ignore it even if it's
                    # in the set of `mutation_ids`
                    yield node
                case Passthrough(children=children) | Container(children=children):
                    rule = self.rules.get(node.symbol)
                    if not isinstance(rule, Grammar.NonTerminal):
                        raise ValueError(
                            "Expected a `NonTerminal` for symbol '{node.symbol}' from the"
                            f" grammar but got rule {rule}"
                        )

                    # If we've already determined the value of this shared symbol
                    if (existing := variables.get(node.symbol)) is not None:
                        yield existing
                        return

                    # If mutate, we return all possible bfs values from that node.
                    if id(node) in mutation_ids:
                        yield from self.bfs(
                            node.symbol,
                            rng_shuffle=rng_shuffle,
                            max_depth=max_mutation_depth,
                            variables=variables,
                        )
                    else:
                        children_itrs: list[Iterator[Node]] = [
                            _inner(c) for c in children
                        ]
                        for new_children in itertools.product(*children_itrs):
                            new_node = node._replace(children=new_children)
                            if rule.shared:
                                variables[new_node.symbol] = new_node
                            yield new_node
                case _:
                    assert_never(node)

        yield from _inner(root)

    def parse(self, s: str) -> Node:  # noqa: C901, PLR0912, PLR0915
        """Parse a `str` into a string of the `Grammar`.

        !!! note

            The initial symbol does not necessarily need to match the
            `start_symbol` of the grammar.

        Args:
            s: the `str` to convert into a string of the `Grammar`.

        Returns:
            The node that represents the string.
        """
        # Chunk up the str
        string_tokens: list[str] = []
        brace_count = 0
        symbol = ""
        for tok in s:
            match tok:
                case " ":
                    continue
                case "(":
                    brace_count += 1
                    if len(symbol) == 0:
                        raise ParseError(
                            f"Opening bracket '(' must be preceeded by symbol"
                            f" but was not.\n{s}"
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

            rule = self.rules.get(symbol)
            match rule:
                case Grammar.Terminal():
                    tokens.append((symbol, self._leafs[symbol]))
                case Grammar.NonTerminal():
                    tokens.append((symbol, rule))
                case None:
                    raise ParseError(
                        f"Invalid symbol '{symbol}', must be either '(', ')', ',' or"
                        f" a symbol in {self.rules.keys()}"
                    )
                case _:
                    assert_never(rule)

        # If we're being strict that shared elements must be the same, then
        # we can do so more cheaply at the beginning by just comparing subtokens
        # before we parse. This will also takes care of subnesting of shared nodes
        # and allow us to skip on some of the token stream as we encounter shared variable
        shared_token_sizes: dict[str, int] = {}
        _shared_locs: dict[str, list[int]] = {s: [] for s in self._shared}

        # We figure out the substrings of where each shared symbol begings and ends
        if _shared_locs:
            bracket_stack: list[int] = []
            bracket_pairs: dict[int, int] = {}
            for i, tok in enumerate(tokens):
                match tok:
                    case "," | (_, Leaf()):
                        continue
                    case ")":
                        start = bracket_stack.pop(-1)
                        bracket_pairs[start] = i
                    case "(":
                        bracket_stack.append(i)
                    case (symbol, Grammar.NonTerminal(shared=shared)):
                        if i + 1 >= len(tokens):
                            raise ParseError(
                                f"Symbol '{tok}' is a `NonTerminal`, implying that it "
                                " should contain some inner elements. However we found it"
                                f" at the last index of the {tokens=}"
                            )
                        if tokens[i + 1] != "(":
                            raise ParseError(
                                f"Symbol '{tok}' at position {i} is a `NonTerminal`,"
                                " implying that it should contain some inner elements."
                                " However it was not followed by a '(' at position"
                                f" {i + 1} in {tokens=}"
                            )
                        if shared is True:
                            _shared_locs[symbol].append(i)
                    case _:
                        assert_never(tok)

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
                        f"First token was symbol '{symbol}' which is a `NonTerminal` that"
                        " is `passthrough`, i.e. it has no associated"
                        " operation and can not be the root."
                    )
                if len(tokens) < 4:
                    raise ParseError(
                        f"First token was symbol '{symbol}' which is"
                        f" a `NonTerminal`, but should have at least 3 more tokens"
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
                        # Re-using a previous one so we can skip ahead in the tokens.
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

    # TODO: The variables thing can mess up the max depth
    def bfs(  # noqa: C901
        self,
        symbol: str,
        *,
        max_depth: int,
        current_depth: int = 0,
        variables: dict[str, Node] | None = None,
        rng_shuffle: np.random.Generator | None = None,
    ) -> Iterator[Node]:
        """Iterate over all possible strings in a breadth first manner.

        Args:
            symbol: The symbol to start the string from.
            max_depth: The maximum depth of the produced string. This may not
                be fully gauranteed given shared `NonTerminal`s. This is required
                to prevent infinite recursion. Any non-terminated strings, i.e. those
                which still require expansion, but have exceeded the depth, will not be
                returned.
            current_depth: What depth this call of the function is acting at. This is used
                recursively and can mostly be left at `0`.
            variables: Any instantiated shared variables used for a `shared=`
                `NonTerminal`.
            rng_shuffle: Whether to shuffle the order of the children when doing breadth
                first search. This may only be required if you are not consuming the full
                iterator this returns. For the most part this can be ignored.

        Returns:
            An iterator over the valid strings in the grammar.
        """
        if current_depth > max_depth:
            return

        variables = variables or {}
        shared_node = variables.get(symbol)
        if shared_node is not None:
            yield shared_node
            return  # TODO: check

        nxt_depth = current_depth + 1

        rule = self.rules.get(symbol)
        match rule:
            case Grammar.Terminal(op=op):
                node = Leaf(symbol, op)
                yield node
            case Grammar.NonTerminal(choices=choices, op=op):
                for choice in choices:
                    children = choice.split(" ")
                    child_expansions: list[Iterator] = [
                        self.bfs(
                            child_symbol,
                            max_depth=max_depth,
                            current_depth=nxt_depth,
                            rng_shuffle=rng_shuffle,
                            variables=variables,
                        )
                        for child_symbol in children
                    ]

                    if rng_shuffle:
                        # Works correctly with python lists, but typing for numpy is off
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
                raise ValueError(f"No symbol {symbol} in rules {self.rules.keys()}")
            case _:
                assert_never(rule)

    def is_valid(
        self,
        node: Node,
        *,
        already_shared: set[str] | None = None,
    ) -> bool:
        """Check if a given string is valid.

        Args:
            node: The start of the string.
            already_shared: Use for recursion, can mostly be kept as `None`.
                Used to ensure that `NonTerminal`s that are `shared=True`, do
                not contain themselves.
        """
        rule = self.rules.get(node.symbol)
        if rule is None:
            raise ValueError(
                f"Node has unknown symbol {node.symbol}, valid symbols are"
                f" {self.rules.keys()}"
            )

        # We should never encounter a situtation where we have some nesting of shared
        # nodes, for example, consider the following, where L2 is shared.
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
                return symbol in self.rules
            case Container(symbol, children, _) | Passthrough(symbol, children):
                s = " ".join(child.symbol for child in children)

                match rule:
                    case Grammar.Terminal(_):
                        return s in self.rules and all(
                            self.is_valid(child, already_shared=already_shared.copy())
                            for child in children
                        )
                    case Grammar.NonTerminal(choices, _):
                        return s in choices and all(
                            self.is_valid(child, already_shared=already_shared.copy())
                            for child in children
                        )
                    case _:
                        assert_never(rule)
                        return None
            case _:
                assert_never(node)
                return None

    def to_model(self, string: str) -> Any:
        """Convert a string form this grammar into its model form."""
        node = self.parse(string)
        return node.to_model()


# TODO: This is just for plotting, not sure where it should go
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
