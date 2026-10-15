"""Log-log power-law fits used to derive the GP mean prior from observed trials."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)


@dataclass
class LogLogPowerLawFit:
    """A data class to hold the results of a log-log power-law fit."""

    slope: float
    intercept: float
    r_squared: float
    slope_std_err: float
    intercept_std_err: float
    num_points: int


def fit_power_law(
    flops: np.ndarray, y_metric: np.ndarray, *, log_scale: bool = True
) -> LogLogPowerLawFit | None:
    """Fit log10(metric) = slope * log10(flops) + intercept with quality stats."""
    fit_x = np.asarray(flops, dtype=float)
    fit_y = np.asarray(y_metric, dtype=float)
    mask = np.isfinite(fit_x) & np.isfinite(fit_y) & (fit_x > 0) & (fit_y > 0)
    fit_x = fit_x[mask]
    fit_y = fit_y[mask]

    if len(fit_x) < 2 or len(np.unique(fit_x)) < 2:
        return None

    if log_scale:
        fit_x = np.log10(fit_x)
        fit_y = np.log10(fit_y)

    result: Any = linregress(fit_x, fit_y)

    return LogLogPowerLawFit(
        slope=float(result.slope),
        intercept=float(result.intercept),
        r_squared=float(result.rvalue**2),
        slope_std_err=float(result.stderr),
        intercept_std_err=float(result.intercept_stderr),
        num_points=len(fit_x),
    )


@dataclass
class PowerLawMeanParams:
    """Power-law GP mean prior (slope/intercept) plus the FLOPs used to derive it."""

    slope: float
    intercept: float
    min_flops: int
    flops: list[float] = field(default_factory=list)
    fit: LogLogPowerLawFit | None = None


def fit_power_law_mean_params(
    configs: Sequence[Mapping[str, Any]],
    metrics: Sequence[float],
    cost_estimator: Callable[..., float],
    default_slope: float = -0.5,
    default_intercept: float = 5.0,
) -> PowerLawMeanParams:
    """Derive the (slope, intercept, min_flops) power-law prior."""
    flops_list: list[float] = []
    metrics_list: list[float] = []
    for config, metric in zip(configs, metrics, strict=False):
        try:
            flop_val = cost_estimator(**config)
        except Exception:
            logger.warning(
                "Failed to estimate FLOPs for config %s: %s",
                config,
                metric,
                exc_info=True,
            )
            continue
        if flop_val > 0 and np.isfinite(flop_val) and np.isfinite(metric):
            flops_list.append(float(flop_val))
            metrics_list.append(float(metric))

    slope, intercept = default_slope, default_intercept
    fit_result = None
    if len(flops_list) >= 2 and len(np.unique(flops_list)) >= 2:
        fit_result = fit_power_law(
            flops=np.array(flops_list),
            y_metric=np.array(metrics_list),
            log_scale=True,
        )
        if fit_result is not None:
            slope, intercept = fit_result.slope, fit_result.intercept

    min_flops = int(min(flops_list)) if flops_list else int(1e16)
    return PowerLawMeanParams(
        slope=slope,
        intercept=intercept,
        min_flops=min_flops,
        flops=flops_list,
        fit=fit_result,
    )
