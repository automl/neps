"""Chinchilla approach 3: a parametric loss `L = E + A/N^alpha + B/D^beta` in the
parameter count N and the data D, see https://arxiv.org/abs/2203.15556.

Minimizing it under `flops = 6 * N * D` gives both the compute optimal model size and
data (`COARSE`) and the loss they reach (`LOSS`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scalefm.prediction.base import Row, last_point_in_curve
from scalefm.prediction.pareto import pareto_front
from scalefm.prediction.scaling_law_fit import (
    ScalingLawParameters,
    get_power_law_curvature,
)

#: Width of serialized `ScalingLawParameters`, see `fit_to_row`.
ROW_WIDTH = 5


@dataclass
class Chinchilla3CoarseInfo:
    """Exponents of the optimum, `N ~ C^a` and `D ~ C^b`."""

    a: float
    b: float


def optimum(
    params: ScalingLawParameters, target_flop: float
) -> tuple[float, float, float]:
    """The `(n_params, n_data, loss)` minimizing the loss at `6 * N * D = target_flop`."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive for Chinchilla prediction")

    alpha = float(params.alpha)
    beta = float(params.beta)
    A = float(params.A)
    B = float(params.B)
    if alpha <= 0 or beta <= 0:
        raise ValueError("Invalid Chinchilla parameters: alpha and beta must be positive")
    if A <= 0 or B <= 0:
        raise ValueError("Invalid Chinchilla parameters: A and B must be positive")

    alpha_plus_beta = alpha + beta
    G = ((alpha * A) / (beta * B)) ** (1.0 / alpha_plus_beta)
    c_div_6 = target_flop / 6.0
    n_target = float(G * (c_div_6 ** (beta / alpha_plus_beta)))
    d_target = float((1.0 / G) * (c_div_6 ** (alpha / alpha_plus_beta)))
    if n_target <= 0 or d_target <= 0:
        raise ValueError("Could not derive positive optimal N and D from Chinchilla fit")

    loss = float(
        params.E
        + params.A / (n_target**params.alpha)
        + params.B / (d_target**params.beta)
    )
    return n_target, d_target, loss


def _valid_rows(rows: list[Row]) -> np.ndarray:
    rows_arr = np.asarray(rows, dtype=float)
    if rows_arr.ndim != 2 or rows_arr.shape[1] != 4:
        raise ValueError("rows must be a 2D array-like with shape (n, 4)")
    mask = np.all(np.isfinite(rows_arr) & (rows_arr > 0), axis=1)
    if np.sum(mask) < 3:
        raise ValueError("Need at least 3 valid rows for Chinchilla curvature fitting")
    return rows_arr[mask]


def fit_loss(
    rows: list[Row],
    *,
    best_loss_guess: float,
    exponent_bounds: tuple[float, float] | None = None,
) -> ScalingLawParameters:
    """Fit the law on the (flops, loss) Pareto front of the rows."""
    valid = _valid_rows(rows)
    n_params, n_tokens, losses, flops = valid.T

    res = pareto_front(losses, flops)
    if not res:
        raise ValueError("Could not compute a Pareto front from the provided rows")
    _pareto_flops, _pareto_losses, pareto_indices = res

    return get_power_law_curvature(
        N=n_params[pareto_indices],
        D=n_tokens[pareto_indices],
        L=losses[pareto_indices],
        best_loss_guess=best_loss_guess,
        exponent_bounds=exponent_bounds,
    )


def fit_coarse(
    rows: list[Row] | list[list[Row]],
    *,
    best_loss_guess: float,
    exponent_bounds: tuple[float, float] | None = None,
) -> ScalingLawParameters:
    """Fit the law on every run, using the final point of grouped learning curves."""
    if rows and isinstance(rows[0], list):
        rows = last_point_in_curve(rows)
    valid = _valid_rows(rows)  # type: ignore[arg-type]
    n_params, n_tokens, losses, _flops = valid.T

    return get_power_law_curvature(
        N=n_params,
        D=n_tokens,
        L=losses,
        best_loss_guess=best_loss_guess,
        exponent_bounds=exponent_bounds,
    )


def predict_loss(
    params: ScalingLawParameters, target_flop: float
) -> tuple[float, ScalingLawParameters]:
    """The loss at the compute optimum for `target_flop`, with the fit as info."""
    _n, _d, loss = optimum(params, target_flop)
    return loss, params


def predict_coarse(
    params: ScalingLawParameters, target_flop: float
) -> tuple[tuple[float, float], Chinchilla3CoarseInfo]:
    """The compute optimal `(n_params, n_data)` for `target_flop`, with its exponents."""
    n, d, _loss = optimum(params, target_flop)
    alpha_plus_beta = float(params.alpha) + float(params.beta)
    return (n, d), Chinchilla3CoarseInfo(
        a=float(params.beta) / alpha_plus_beta, b=float(params.alpha) / alpha_plus_beta
    )


def fit_to_row(params: ScalingLawParameters | None) -> list[float]:
    """Serialize a fit to a fixed-width row; a failed fit becomes an all-NaN row."""
    if params is None:
        return [float("nan")] * ROW_WIDTH
    return [params.E, params.A, params.B, params.alpha, params.beta]


def fit_from_row(row: np.ndarray) -> ScalingLawParameters | None:
    """Deserialize a fit row, None for a failed fit.

    A row with any NaN is a failed fit, since the closed-form optimum needs every field.
    """
    row = np.asarray(row, dtype=float)
    if np.any(np.isnan(row)):
        return None
    E, A, B, alpha, beta = row
    return ScalingLawParameters(
        E=float(E), A=float(A), B=float(B), alpha=float(alpha), beta=float(beta)
    )
