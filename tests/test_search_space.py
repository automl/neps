from __future__ import annotations

import pytest

from neps import Categorical, Constant, Float, Grammar, Integer, SearchSpace


def test_search_space_orders_parameters_by_name():
    unsorted = SearchSpace({"b": Float(0, 1), "c": Float(0, 1), "a": Float(0, 1)})
    expected = SearchSpace({"a": Float(0, 1), "b": Float(0, 1), "c": Float(0, 1)})
    assert unsorted == expected


def test_multipe_fidelities_raises_error():
    # We should allow this at some point, but until we do, raise an error
    with pytest.raises(ValueError, match="neps only supports one fidelity parameter"):
        SearchSpace(
            {"a": Float(0, 1, is_fidelity=True), "b": Float(0, 1, is_fidelity=True)}
        )


def test_mutliple_grammars_raises_error():
    with pytest.raises(ValueError, match="neps only supports one grammar parameter"):
        SearchSpace(
            {
                "a": Grammar.from_dict("s", {"s": lambda _: None}),
                "b": Grammar.from_dict("s", {"s": lambda _: None}),
            }
        )


def test_sorting_of_parameters_into_subsets():
    elements = {
        "a": Float(0, 1),
        "b": Integer(0, 10),
        "c": Categorical(["a", "b", "c"]),
        "d": Float(0, 1, is_fidelity=True),
        "x": Constant("x"),
        "g": Grammar.from_dict("s", {"s": lambda _: None}),
    }
    space = SearchSpace(elements)
    assert space.elements == elements
    assert space.categoricals == {"c": elements["c"]}
    assert space.numerical == {"a": elements["a"], "b": elements["b"]}
    assert space.fidelities == {"d": elements["d"]}
    assert space.constants == {"x": "x"}
    assert space.grammars == {"g": elements["g"]}

    parameters = {**space.numerical, **space.categoricals}
    assert parameters == {
        "a": elements["a"],
        "b": elements["b"],
        "c": elements["c"],
    }
    assert space.fidelity == ("d", elements["d"])
    assert space.grammar == ("g", elements["g"])
