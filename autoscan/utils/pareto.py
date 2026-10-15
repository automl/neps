"""Pareto dominance, where every objective is minimized."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether `a` is at least as good as `b` in every objective and better in one."""
    return all(ai <= bi for ai, bi in zip(a, b, strict=True)) and any(
        ai < bi for ai, bi in zip(a, b, strict=True)
    )


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
