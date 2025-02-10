from __future__ import annotations

from dataclasses import dataclass

import pytest
from graph import (
    Container,
    Grammar,
    Leaf,
    Node,
    ParseError,
    Passthrough,
    bfs_node,
    dfs_node,
    parse,
    select,
    to_model,
    to_node_from_graph,
    to_nxgraph,
    to_string,
)


# Leafs
@dataclass
class T:
    s: str

    # This is the `op()`
    def __call__(self) -> str:
        return self.s


def join(*s: str) -> str:
    return "[" + "".join(s) + "]"


grammar_1 = Grammar.from_dict(
    {
        "s": (["a", "b", "p", "p p"], join),
        "p": ["a b", "s"],
        "a": T("a"),
        "b": T("b"),
    }
)

grammar_2 = Grammar.from_dict(
    {
        "L1": (["L2 L2 L3"], join),
        "L2": Grammar.NonTerminal(["L3"], join, shared=True),
        "L3": Grammar.NonTerminal(["a", "b"], None, shared=True),
        "a": T("a"),
        "b": T("a"),
    }
)


@pytest.mark.parametrize(
    ("grammar", "string", "built", "node"),
    [
        (grammar_1, "a", "a", Leaf("a", T("a"))),
        (grammar_1, "b", "b", Leaf("b", T("b"))),
        (
            grammar_1,
            "s(a)",
            "[a]",
            Container("s", op=join, children=[Leaf("a", T("a"))]),
        ),
        (
            grammar_1,
            "s(p(a, b))",
            "[ab]",
            Container(
                "s",
                children=[
                    Passthrough(
                        "p",
                        children=[Leaf("a", T("a")), Leaf("b", T("b"))],
                    ),
                ],
                op=join,
            ),
        ),
        (
            grammar_1,
            "s(p(a, b), p(s(a)))",
            "[ab[a]]",
            Container(
                "s",
                children=[
                    Passthrough(
                        "p",
                        children=[Leaf("a", T("a")), Leaf("b", T("b"))],
                    ),
                    Passthrough(
                        "p",
                        children=[Container("s", children=[Leaf("a", T("a"))], op=join)],
                    ),
                ],
                op=join,
            ),
        ),
        (
            grammar_1,
            "s(p(s(a)))",
            "[[a]]",
            Container(
                "s",
                children=[
                    Passthrough(
                        "p",
                        children=[
                            Container(
                                "s",
                                children=[Leaf("a", T("a"))],
                                op=join,
                            )
                        ],
                    ),
                ],
                op=join,
            ),
        ),
    ],
)
def test_string_serialization_and_deserialization_correct(
    grammar: Grammar,
    string: str,
    built: str,
    node: Node,
) -> None:
    # Test parsing
    parsed = parse(grammar, string)
    assert parsed == node

    # Test serialization
    serialized_again = to_string(parsed)
    assert serialized_again == string

    # Test building
    assert to_model(parsed) == built

    # Test graph and back again
    graph = to_nxgraph(parsed, include_passthroughs=True)
    node_again = to_node_from_graph(graph, grammar)
    assert parsed == node_again


@pytest.mark.parametrize(
    ("grammar", "string"),
    [
        (grammar_1, "c"),
        (grammar_1, ""),
        (grammar_1, "s(a"),
        (grammar_1, "p(a, b)"),
        (grammar_1, "("),
        (grammar_1, "s(a))"),
        (grammar_1, "s((a)"),
        (grammar_1, "s("),
        (grammar_1, "s)"),
        (grammar_1, "a, a"),
        (grammar_1, "a,"),
        (grammar_1, "s, s"),
        # Invalid due to shared rule but not sharing values
        (grammar_2, "L1(L2(L3(a)), L2(L3(a)), L3(b))"),
    ],
)
def test_string_deserialization_fail_cases(grammar: Grammar, string: str) -> None:
    with pytest.raises(ParseError):
        parse(grammar, string)


def test_dfs_node_container() -> None:
    node = Container(
        "s",
        children=[
            Container(
                "s_left",
                children=[Leaf("a_left", T("a")), Leaf("b_left", T("b"))],
                op=join,
            ),
            Container(
                "s_right",
                children=[Leaf("a_right", T("a")), Leaf("b_right", T("b"))],
                op=join,
            ),
        ],
        op=join,
    )
    outcome = list(dfs_node(node))
    expected = [
        # First
        Container(
            "s",
            children=[
                Container(
                    "s_left",
                    children=[Leaf("a_left", T("a")), Leaf("b_left", T("b"))],
                    op=join,
                ),
                Container(
                    "s_right",
                    children=[Leaf("a_right", T("a")), Leaf("b_right", T("b"))],
                    op=join,
                ),
            ],
            op=join,
        ),
        # go down left depth first
        Container(
            "s_left",
            children=[Leaf("a_left", T("a")), Leaf("b_left", T("b"))],
            op=join,
        ),
        Leaf("a_left", T("a")),
        Leaf("b_left", T("b")),
        # go down right depth first
        Container(
            "s_right",
            children=[Leaf("a_right", T("a")), Leaf("b_right", T("b"))],
            op=join,
        ),
        Leaf("a_right", T("a")),
        Leaf("b_right", T("b")),
    ]
    for i, (e, o) in enumerate(zip(expected, outcome, strict=True)):
        assert e == o, f"Failed at index {i}"


def test_bfs_node_container() -> None:
    node = Container(
        "s",
        children=[
            Container(
                "s_left",
                children=[Leaf("a_left", T("a")), Leaf("b_left", T("b"))],
                op=join,
            ),
            Container(
                "s_right",
                children=[Leaf("a_right", T("a")), Leaf("b_right", T("b"))],
                op=join,
            ),
        ],
        op=join,
    )
    outcome = list(bfs_node(node))
    expected = [
        # First
        Container(
            "s",
            children=[
                Container(
                    "s_left",
                    children=[Leaf("a_left", T("a")), Leaf("b_left", T("b"))],
                    op=join,
                ),
                Container(
                    "s_right",
                    children=[Leaf("a_right", T("a")), Leaf("b_right", T("b"))],
                    op=join,
                ),
            ],
            op=join,
        ),
        # Second level first
        Container(
            "s_left",
            children=[Leaf("a_left", T("a")), Leaf("b_left", T("b"))],
            op=join,
        ),
        Container(
            "s_right",
            children=[Leaf("a_right", T("a")), Leaf("b_right", T("b"))],
            op=join,
        ),
        # Then 3rd level
        Leaf("a_left", T("a")),
        Leaf("b_left", T("b")),
        Leaf("a_right", T("a")),
        Leaf("b_right", T("b")),
    ]
    for i, (e, o) in enumerate(zip(expected, outcome, strict=True)):
        assert e == o, f"Failed at index {i}"


def test_select_symbol() -> None:
    root = Container(
        "a",
        children=[
            Container(
                "b",
                children=[
                    Container(
                        "d",
                        children=[Leaf("l1", op=T("l1"))],
                        op=join,
                    ),
                ],
                op=join,
            ),
            Container("c", children=[Leaf("l2", op=T("l2"))], op=join),
            Leaf("l3", op=T("l3")),
            Container(
                "d",
                children=[Leaf("l4", op=T("l4"))],
                op=join,
            ),
        ],
        op=join,
    )
    selected = list(select(root, how=("symbol", "d")))
    assert selected == [
        Container(
            "d",
            children=[Leaf("l4", op=T("l4"))],
            op=join,
        ),
        Container(
            "d",
            children=[Leaf("l1", op=T("l1"))],
            op=join,
        ),
    ]


def test_select_depth() -> None:
    root = Container(
        "a",
        children=[
            Container(
                "b",
                children=[
                    Container(
                        "d",
                        children=[Leaf("l1", op=T("l1"))],
                        op=join,
                    ),
                ],
                op=join,
            ),
            Container("c", children=[Leaf("l2", op=T("l2"))], op=join),
            Leaf("l3", op=T("l3")),
            Container(
                "d",
                children=[Leaf("l4", op=T("l4"))],
                op=join,
            ),
        ],
        op=join,
    )
    selected = list(select(root, how=("depth", 1)))
    assert selected == root.children

    selected = list(select(root, how=("depth", range(1, 3))))
    expected = [
        # Depth 1
        *root.children,
        # Depth 2
        Container(
            "d",
            children=[Leaf("l1", op=T("l1"))],
            op=join,
        ),
        Leaf("l2", op=T("l2")),
        Leaf("l4", op=T("l4")),
    ]
    assert selected == expected


def test_select_climb() -> None:
    # NOTE: The order is rather arbitrary and not much thought has been given to it.
    # However the test still tests a particular order that was done by trial and
    # error. Feel free to redo the order if this changes.
    root = Container(
        "a",
        children=[
            Container(
                "b",
                children=[
                    Container(
                        "d",
                        children=[Leaf("l1", op=T("l1"))],
                        op=join,
                    ),
                ],
                op=join,
            ),
            Container("c", children=[Leaf("l2", op=T("l2"))], op=join),
            Leaf("l3", op=T("l3")),
            Container(
                "d",
                children=[Leaf("l4", op=T("l4"))],
                op=join,
            ),
        ],
        op=join,
    )
    selected = list(select(root, how=("climb", 0)))
    assert selected == [
        Leaf("l3", op=T("l3")),
        Leaf("l2", op=T("l2")),
        Leaf("l4", op=T("l4")),
        Leaf("l1", op=T("l1")),
    ]

    selected = list(select(root, how=("climb", range(1, 3))))
    expected = [
        root,
        Container("c", children=[Leaf("l2", op=T("l2"))], op=join),
        Container(
            "d",
            children=[Leaf("l4", op=T("l4"))],
            op=join,
        ),
        Container(
            "d",
            children=[Leaf("l1", op=T("l1"))],
            op=join,
        ),
        Container(
            "b",
            children=[
                Container(
                    "d",
                    children=[Leaf("l1", op=T("l1"))],
                    op=join,
                ),
            ],
            op=join,
        ),
    ]
    for i, (sel, exp) in enumerate(zip(selected, expected, strict=True)):
        assert sel == exp, f"Mismatch at pos {i}:\nExpected: {exp}\n\nGot: {sel}"
