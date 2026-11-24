"""Pareto efficiency, where every objective is minimized."""

from __future__ import annotations

import numpy as np


def pareto_efficient(X: np.ndarray) -> np.ndarray:
    """Evaluates for each allocation whether it is Pareto efficient.

    Args:
        X: The allocations to check, of shape `(N, D)` for `N` allocations with `D`
            costs each.

    Returns:
        A boolean array of shape `(N,)`, whether each allocation is Pareto efficient.
    """
    mask = np.ones(X.shape[0], dtype=bool)
    for i, allocation in enumerate(X):
        if mask[i]:
            dominated = np.all(allocation <= X[mask], axis=1) * np.any(
                allocation < X[mask], axis=1
            )
            mask[mask] = ~dominated

    return mask
