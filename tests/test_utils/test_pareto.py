from __future__ import annotations

import numpy as np

from extrapolaw.utils.pareto import pareto_efficient


def test_pareto_efficient() -> None:
    # (flops, loss): duplicates of a front point are kept, equal flops keep the best
    points = np.array(
        [[1.0, 5.0], [2.0, 3.0], [2.0, 4.0], [3.0, 3.0], [4.0, 1.0], [4.0, 1.0]]
    )
    assert pareto_efficient(points).tolist() == [True, True, False, False, True, True]
    assert pareto_efficient(np.empty((0, 2))).tolist() == []
