"""Pareto fronts over (flops, loss), both minimized, and search metrics built on them."""

from __future__ import annotations

import itertools
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scalefm.prediction.scaling_law_fit import LogLogPowerLawFit, fit_power_law

#: Reference point of the hypervolume, just outside the normalized unit square so
#: the worst valid model still contributes a non-zero area.
HYPERVOLUME_REFERENCE = (1.1, 1.1)


def non_dominated_mask(flops: Any, losses: Any) -> np.ndarray:
    """Boolean mask of the points no other point beats on both flops and loss.

    Inputs must be finite. Exact duplicates of a front point are all kept.
    """
    flops = np.asarray(flops, dtype=float)
    losses = np.asarray(losses, dtype=float)
    mask = np.zeros(len(flops), dtype=bool)
    if len(flops) == 0:
        return mask

    order = np.lexsort((losses, flops))
    sorted_flops, sorted_losses = flops[order], losses[order]
    # Sorted by loss within equal flops, so each group starts with its best loss.
    group_start = np.r_[True, sorted_flops[1:] != sorted_flops[:-1]]
    group = np.cumsum(group_start) - 1
    group_best = sorted_losses[group_start]
    best_at_smaller_flops = np.r_[np.inf, np.minimum.accumulate(group_best)[:-1]]

    mask[order] = (sorted_losses == group_best[group]) & (
        sorted_losses < best_at_smaller_flops[group]
    )
    return mask


def pareto_front(
    losses: Any, flops: Any
) -> tuple[list[float], list[float], list[int]] | None:
    """Pareto front of (flops, loss), sorted by flops.

    Returned as (flops, losses, indices into the inputs).

    Non-finite points are ignored. Returns None when no usable point is left.
    """
    flops = np.asarray(flops, dtype=float)
    losses = np.asarray(losses, dtype=float)
    if len(flops) != len(losses):
        return None

    valid = np.flatnonzero(np.isfinite(flops) & np.isfinite(losses))
    if len(valid) == 0:
        return None

    front = valid[non_dominated_mask(flops[valid], losses[valid])]
    front = front[np.argsort(flops[front], kind="stable")]
    return flops[front].tolist(), losses[front].tolist(), front.tolist()


def enforce_monotonic_params(rows: list[tuple]) -> list[tuple]:
    """Drop (n_params, n_data, loss, flops) rows whose n_params rises as flops falls.

    Walks down from the largest-flops row; returns the kept rows sorted by ascending
    flops.
    """
    if len(rows) <= 1:
        return rows

    descending = sorted(rows, key=lambda row: row[3], reverse=True)
    monotonic = [descending[0]]
    for row in descending[1:]:
        if row[0] <= monotonic[-1][0]:
            monotonic.append(row)
    return monotonic[::-1]


class _RunningFront:
    """Pareto front of the points added so far.

    Kept sorted by ascending x, so by descending y.
    """

    def __init__(self) -> None:
        self.xs: list[float] = []
        self.ys: list[float] = []

    def add(self, x: float, y: float) -> bool:
        """Add a point; returns whether the front changed.

        A repeat of a front point changes nothing.
        """
        i = bisect_left(self.xs, x)
        if (i > 0 and self.ys[i - 1] <= y) or (
            i < len(self.xs) and self.xs[i] == x and self.ys[i] <= y
        ):
            return False

        j = i
        while j < len(self.xs) and self.ys[j] >= y:
            j += 1
        self.xs[i:j] = [x]
        self.ys[i:j] = [y]
        return True


def _search_history(
    losses: Any, accumulated_flops: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Usable (losses, accumulated flops, per-trial flops) of a chronological search."""
    losses = np.asarray(losses, dtype=float)
    accumulated = np.asarray(accumulated_flops, dtype=float)
    if len(losses) == 0 or len(losses) != len(accumulated):
        return None

    valid = np.isfinite(losses) & np.isfinite(accumulated) & (accumulated >= 0)
    if not np.any(valid):
        return None
    losses, accumulated = losses[valid], accumulated[valid]
    return losses, accumulated, np.diff(accumulated, prepend=0.0)


def hypervolume_over_search(
    losses: Any,
    accumulated_flops: Any,
    loss_bounds: tuple[float, float] | None = None,
    flops_bounds: tuple[float, float] | None = None,
) -> tuple[list[float], list[float]] | None:
    """Hypervolume of the search's Pareto front after each trial.

    Returned as (accumulated flops, hypervolumes).

    Loss and log10(per-trial flops) are scaled to the unit square with `loss_bounds`
    and `flops_bounds` (log10 flops); pass bounds shared by all compared runs, since
    the defaults are inferred from this run alone. Higher is better.
    """
    history = _search_history(losses, accumulated_flops)
    if history is None:
        return None
    losses, accumulated, trial_flops = history
    log_flops = np.log10(np.maximum(1.0, trial_flops))

    min_loss, max_loss = loss_bounds or (losses.min(), losses.max())
    min_flops, max_flops = flops_bounds or (log_flops.min(), log_flops.max())
    norm_losses = (losses - min_loss) / max(1e-9, max_loss - min_loss)
    norm_flops = (log_flops - min_flops) / max(1e-9, max_flops - min_flops)

    ref_x, ref_y = HYPERVOLUME_REFERENCE
    front = _RunningFront()
    hypervolumes: list[float] = []
    hv = 0.0
    for x, y in zip(norm_flops.tolist(), norm_losses.tolist(), strict=True):
        if front.add(x, y):
            hv = 0.0
            for k, (front_x, front_y) in enumerate(zip(front.xs, front.ys, strict=True)):
                next_x = front.xs[k + 1] if k + 1 < len(front.xs) else ref_x
                hv += (next_x - front_x) * (ref_y - front_y)
            hv = min(hv, ref_x * ref_y)
            if hypervolumes:
                hv = max(hv, hypervolumes[-1])
        hypervolumes.append(hv)

    return accumulated.tolist(), hypervolumes


def _area_between_lines(
    a: LogLogPowerLawFit, b: LogLogPowerLawFit, x_left: float, x_right: float
) -> float:
    """Integral of the absolute vertical distance between two fitted lines.

    Integrated over [x_left, x_right].
    """
    slope = a.slope - b.slope
    intercept = a.intercept - b.intercept

    edges = [x_left, x_right]
    if abs(slope) > 1e-12 and x_left < -intercept / slope < x_right:
        edges.insert(1, -intercept / slope)

    area = 0.0
    for start, stop in itertools.pairwise(edges):
        sign = 1.0 if slope * 0.5 * (start + stop) + intercept >= 0.0 else -1.0
        area += sign * (0.5 * slope * (stop**2 - start**2) + intercept * (stop - start))
    return max(0.0, area)


def pareto_line_area_to_true(
    losses: Any,
    accumulated_flops: Any,
    true_pareto_info: Mapping[str, Sequence[float]],
) -> tuple[list[float], list[float]] | None:
    """Area between the search's fitted Pareto line and the true one after each trial.

    Both lines are power-law fits in log10(loss) vs log10(per-trial flops), integrated
    over the true front's flops range. Nothing is normalized, so the value does not
    depend on which other runs it is compared with. Lower is better; NaN while the
    search front has too few points to fit. Returns (accumulated flops, areas).
    """
    true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
    true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
    usable = (
        np.isfinite(true_flops)
        & np.isfinite(true_losses)
        & (true_flops > 0)
        & (true_losses > 0)
    )
    if np.sum(usable) < 2:
        return None

    history = _search_history(losses, accumulated_flops)
    if history is None:
        return None
    losses, accumulated, trial_flops = history
    log_losses = np.log10(np.maximum(1e-12, losses))
    log_flops = np.log10(np.maximum(1.0, trial_flops))

    true_flops = np.maximum(1.0, true_flops[usable])
    true_losses = np.maximum(1e-12, true_losses[usable])
    on_front = non_dominated_mask(true_flops, true_losses)
    true_fit = fit_power_law(true_flops[on_front], true_losses[on_front])
    if true_fit is None:
        return None
    x_left = float(np.log10(true_flops[on_front].min()))
    x_right = float(np.log10(true_flops[on_front].max()))

    front = _RunningFront()
    areas: list[float] = []
    area = float("nan")
    for x, y in zip(log_flops.tolist(), log_losses.tolist(), strict=True):
        if front.add(x, y):
            fit = fit_power_law(
                10.0 ** np.asarray(front.xs), 10.0 ** np.asarray(front.ys)
            )
            area = (
                float("nan")
                if fit is None
                else _area_between_lines(fit, true_fit, x_left, x_right)
            )
        areas.append(area)

    return accumulated.tolist(), areas
