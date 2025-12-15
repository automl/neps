from __future__ import annotations

import pytest

from neps.space import SearchSpace
from neps.space.parameters import HPOCategorical, HPOConstant, HPOFloat, HPOInteger


def test_search_space_orders_parameters_by_name():
    unsorted = SearchSpace(
        {"b": HPOFloat(0, 1), "c": HPOFloat(0, 1), "a": HPOFloat(0, 1)}
    )
    expected = SearchSpace(
        {"a": HPOFloat(0, 1), "b": HPOFloat(0, 1), "c": HPOFloat(0, 1)}
    )
    assert unsorted == expected


def test_multipe_fidelities_raises_error():
    # We should allow this at some point, but until we do, raise an error
    with pytest.raises(ValueError, match="neps only supports one fidelity parameter"):
        SearchSpace(
            {
                "a": HPOFloat(0, 1, is_fidelity=True),
                "b": HPOFloat(0, 1, is_fidelity=True),
            }
        )


def test_sorting_of_parameters_into_subsets():
    elements = {
        "a": HPOFloat(0, 1),
        "b": HPOInteger(0, 10),
        "c": HPOCategorical(["a", "b", "c"]),
        "d": HPOFloat(0, 1, is_fidelity=True),
        "x": HPOConstant("x"),
    }
    space = SearchSpace(elements)
    assert space.elements == elements
    assert space.categoricals == {"c": elements["c"]}
    assert space.numerical == {"a": elements["a"], "b": elements["b"]}
    assert space.fidelities == {"d": elements["d"]}
    assert space.constants == {"x": "x"}

    assert space.searchables == {
        "a": elements["a"],
        "b": elements["b"],
        "c": elements["c"],
    }
    assert space.fidelity == ("d", elements["d"])
