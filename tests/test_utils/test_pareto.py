from __future__ import annotations

import numpy as np
import pytest

from autoscan.utils.pareto import dominates, pareto_efficient


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0, 1.0], [2.0, 2.0], True),
        ([1.0, 2.0], [1.0, 3.0], True),
        ([1.0, 2.0], [1.0, 2.0], False),  # Equal, so not better in any objective
        ([1.0, 3.0], [2.0, 2.0], False),  # A trade-off
        ([2.0], [1.0], False),
        ([1.0], [2.0], True),
    ],
)
def test_dominates(a: list[float], b: list[float], expected: bool) -> None:
    assert dominates(a, b) is expected


def test_pareto_efficient() -> None:
    # (flops, loss): duplicates of a front point are kept, equal flops keep the best
    points = np.array(
        [[1.0, 5.0], [2.0, 3.0], [2.0, 4.0], [3.0, 3.0], [4.0, 1.0], [4.0, 1.0]]
    )
    assert pareto_efficient(points).tolist() == [True, True, False, False, True, True]
    assert pareto_efficient(np.empty((0, 2))).tolist() == []
