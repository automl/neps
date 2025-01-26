from __future__ import annotations

import pytest

from neps import Categorical, Constant, Float, Integer, SearchSpace


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


def test_sorting_of_parameters_into_subsets():
    elements = {
        "a": Float(0, 1),
        "b": Integer(0, 10),
        "c": Categorical(["a", "b", "c"]),
        "d": Float(0, 1, is_fidelity=True),
        "x": Constant("a"),
    }
    space = SearchSpace(elements)
    assert space.elements == elements
    assert space.categoricals == {"c": elements["c"]}
    assert space.numerical == {"a": elements["a"], "b": elements["b"]}
    assert space.fidelities == {"d": elements["d"]}
    assert space.constants == {"x": elements["x"]}

    assert space.searchables == {
        "a": elements["a"],
        "b": elements["b"],
        "c": elements["c"],
    }
    assert space.fidelity == ("d", elements["d"])
