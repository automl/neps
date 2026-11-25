"""Fits of power laws and of the `L = E + A/N^alpha + B/D^beta` scaling law."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from scalefm.utils.power_law import (
    LogLogPowerLawFit,
    PowerLawMeanParams,
    fit_power_law,
    fit_power_law_mean_params,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_EXPONENT_BOUNDS",
    "LogLogPowerLawFit",
    "PowerLawMeanParams",
    "ScalingLawParameters",
    "fit_power_law",
    "fit_power_law_mean_params",
    "get_power_law_curvature",
    "predict_from_loglog_fit",
]


def predict_from_loglog_fit(
    fit: LogLogPowerLawFit,
    target_flop: float,
    log_scale: bool = True,  # noqa: FBT001, FBT002
    z_score: float = 1.96,
) -> tuple[float, float, float]:
    """Predict loss at target FLOPs and return mean/upper/lower (95% CI default)."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive for log-log prediction")

    fitted_y = float(np.log10(target_flop)) if log_scale else float(target_flop)

    predicted_y = (fit.slope * fitted_y) + fit.intercept
    if log_scale:
        predicted_y = float(np.power(10.0, predicted_y))

    y_upper = (fit.slope + (z_score * fit.slope_std_err)) * fitted_y + (
        fit.intercept + (z_score * fit.intercept_std_err)
    )
    y_lower = (fit.slope - (z_score * fit.slope_std_err)) * fitted_y + (
        fit.intercept - (z_score * fit.intercept_std_err)
    )
    if log_scale:
        y_upper = float(np.power(10.0, y_upper))
        y_lower = float(np.power(10.0, y_lower))
    return predicted_y, y_upper, y_lower


@dataclass
class ScalingLawParameters:
    """Coefficients of the scaling law `L = E + A/N^alpha + B/D^beta`."""

    E: float
    A: float
    B: float
    alpha: float
    beta: float


#: Default search range for the alpha/beta exponents of `L = E + A/N^a + B/D^b`.
#: Appropriate for a next-token/contrastive training loss, which falls fast
#: enough in N and D to land inside it. Accuracy-like targets fall far more
#: slowly and need a wider range -- see `get_power_law_curvature`.
DEFAULT_EXPONENT_BOUNDS = (0.25, 0.75)


def get_power_law_curvature(  # noqa: C901, PLR0915
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    best_loss_guess: float,
    exponent_bounds: tuple[float, float] | None = None,
) -> ScalingLawParameters:
    """Fit `L = E + A/N^alpha + B/D^beta` by Huber loss in log space.

    `best_loss_guess` upper-bounds the irreducible term E (searched over
    `(best_loss_guess * 1e-3, best_loss_guess)`), so it must be on the scale
    of whatever `L` is -- for an accuracy-like target extracted as `1 - acc`
    that is `1 - <achievable accuracy ceiling>`, not the training loss floor.

    `exponent_bounds` bounds alpha and beta; defaults to
    `DEFAULT_EXPONENT_BOUNDS`. Passing a range that excludes the true
    exponents does not merely widen the error bars: both exponents rail at
    the same bound, and the derived compute-optimal exponent
    `beta / (alpha + beta)` collapses to exactly 0.5 independent of the data.
    """
    lo, hi = DEFAULT_EXPONENT_BOUNDS if exponent_bounds is None else exponent_bounds
    if not 0 < lo < hi:
        raise ValueError(f"exponent_bounds must satisfy 0 < lo < hi, got {(lo, hi)}")

    def huber(residual: np.ndarray, delta: float = 1e-3) -> np.ndarray:
        abs_r = np.abs(residual)
        return np.where(
            abs_r <= delta,
            0.5 * residual**2,
            delta * (abs_r - 0.5 * delta),
        )

    def objective(
        theta: np.ndarray, N: np.ndarray, D: np.ndarray, L: np.ndarray
    ) -> float:
        E, a, b, alpha, beta = theta

        # WIDENED clipping to allow negative exponents for values < 1.0
        a_clipped = np.clip(a, -20, 20)
        b_clipped = np.clip(b, -20, 20)
        A = np.exp(a_clipped)
        B = np.exp(b_clipped)

        with np.errstate(divide="ignore", invalid="ignore"):
            N_alpha = np.power(
                N, alpha, where=(N > 0), out=np.ones_like(N, dtype=np.float64)
            )
            D_beta = np.power(
                D, beta, where=(D > 0), out=np.ones_like(D, dtype=np.float64)
            )

            N_alpha = np.where(N_alpha > 0, N_alpha, 1.0)
            D_beta = np.where(D_beta > 0, D_beta, 1.0)

            L_hat = E + A / N_alpha + B / D_beta

        if np.any(~np.isfinite(L_hat)) or np.any(L_hat <= 0):
            return 1e10

        residual = np.log(L_hat) - np.log(L)
        loss = np.sum(huber(residual))

        if not np.isfinite(loss):
            return 1e10

        return float(loss)

    num_points = len(L)
    min_L = best_loss_guess

    # Multi-start values for alpha/beta. The literals below reproduce the
    # historical default-bounds behaviour exactly; a custom range gets a
    # geometric ladder across it instead, since 0.4/0.5 can sit outside it.
    if (lo, hi) == DEFAULT_EXPONENT_BOUNDS:
        exp_starts_small = [0.5]
        exp_starts_grid = [0.4, 0.5]
    else:
        exp_starts_small = [float(np.sqrt(lo * hi))]
        exp_starts_grid = [float(v) for v in np.geomspace(lo, hi, 4)[:3]]

    # FIXED: Initialize a and b closer to 0 or negative for normalized data
    # exp(0) = 1.0, exp(-2) = 0.13. These are realistic scales for normalized A and B.
    if num_points <= 10:
        initialization = [
            (min_L * e_scale, a0, a0, exp0, exp0)
            for e_scale, a0 in ((0.9, 0), (0.8, -2), (0.7, 2))
            for exp0 in exp_starts_small
        ]
    else:
        initialization = list(
            itertools.product(
                [min_L * 0.8, min_L * 0.9],
                [-2, 0, 2],  # a
                [-2, 0, 2],  # b
                exp_starts_grid,  # alpha
                exp_starts_grid,  # beta
            )
        )

    best_result: Any = None
    best_loss = float("inf")

    N_scale = np.max(N)
    D_scale = np.max(D)

    N_norm = N / N_scale
    D_norm = D / D_scale

    for theta0 in initialization:
        try:
            # FIXED: Allow a and b to be negative, up to -20 (A = 2e-9)
            bounds = [
                (min_L * 0.001, min_L),
                (-20, 20),  # a
                (-20, 20),  # b
                (lo, hi),  # alpha
                (lo, hi),  # beta
            ]

            res = minimize(
                objective,
                theta0,
                args=(N_norm, D_norm, L),
                method="L-BFGS-B",
                bounds=bounds,
            )

            # FIXED: Removed the strict `res.success` check.
            # If SciPy throws a line search warning but finds a valid lower loss, keep it!
            if np.isfinite(res.fun) and res.fun < best_loss:
                best_loss = res.fun
                best_result = res

        except Exception:
            logger.debug("Scaling law fit failed from start %s", theta0, exc_info=True)
            continue

    if best_result is None:
        raise ValueError("Could not fit scaling law with any initialization.")

    E, a, b, alpha, beta = best_result.x

    # Apply the same widened clipping before undoing the log
    A = np.exp(np.clip(a, -20, 20))
    B = np.exp(np.clip(b, -20, 20))

    # Correctly un-normalize A and B back to raw compute space
    A = A * (N_scale**alpha)
    B = B * (D_scale**beta)

    logger.debug(
        f"Fitted scaling law parameters: E={E:.6f}, A={A:.6e}, B={B:.6e},"
        f" alpha={alpha:.4f}, beta={beta:.4f}"
    )
    return ScalingLawParameters(E=E, A=A, B=B, alpha=alpha, beta=beta)
