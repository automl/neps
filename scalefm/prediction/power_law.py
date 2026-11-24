"""Power-law prediction: log-log linear fits in the compute, one per predicted quantity.

* `LOSS`: the loss on the (flops, loss) Pareto front, see https://arxiv.org/abs/2001.08361.
* `COARSE`: the model size on the learning-curve envelope (Chinchilla approach 1),
  with `n_data = flops / (6 * n_params)`.
* `FULL`: every searchable hyperparameter on the (flops, loss) Pareto front.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.ndimage import gaussian_filter1d

from scalefm.prediction.base import Row, flatten_rows
from scalefm.prediction.pareto import pareto_front
from scalefm.prediction.scaling_law_fit import (
    LogLogPowerLawFit,
    fit_power_law,
    predict_from_loglog_fit,
)
from scalefm.space import HPOFloat, HPOInteger, SearchSpace

#: Chinchilla approach 1 reads the learning-curve envelope at this many log-spaced
#: flops values.
LC_ENVELOPE_NUM_FLOPS = 1500
#: Width of the Gaussian smoothing applied to each interpolated learning curve, in
#: log10(flops).
LC_SMOOTHING_SIGMA = 0.05

#: Width of a serialized `LogLogPowerLawFit`, see `fit_to_row`.
ROW_WIDTH = 6


@dataclass
class PowerLawLossInfo:
    """The loss power law fit and its prediction at one compute budget."""

    slope: float
    intercept: float
    r_squared: float
    slope_std_err: float
    num_pareto_points: int
    predicted_loss: float
    predicted_loss_upper: float
    predicted_loss_lower: float


@dataclass
class PowerLawCoarseInfo:
    """The model size power law fit behind a coarse prediction."""

    slope: float
    intercept: float
    r_squared: float
    slope_std_err: float


@dataclass
class PowerLawFullFit:
    """One power law per searchable HP, plus the Pareto front it was fitted on."""

    hp_fits: dict[str, LogLogPowerLawFit]
    pareto_ids: list[int]
    pareto_configs: list[dict[str, Any]]
    pareto_flops: list[float]
    pareto_losses: list[float]


# --- LOSS ------------------------------------------------------------------------


def fit_loss(rows: list[Row]) -> LogLogPowerLawFit:
    """Fit log10(loss) against log10(flops) on the (flops, loss) Pareto front."""
    rows_arr = np.asarray(rows, dtype=float)
    if rows_arr.ndim != 2 or rows_arr.shape[1] != 4:
        raise ValueError("rows must be a 2D array-like with shape (n, 4)")

    losses_arr = rows_arr[:, 2]
    flops_arr = rows_arr[:, 3]
    mask = (
        np.isfinite(losses_arr)
        & np.isfinite(flops_arr)
        & (losses_arr > 0)
        & (flops_arr > 0)
    )
    if not np.any(mask):
        raise ValueError("rows do not contain any finite positive loss/flop values")

    res = pareto_front(losses_arr[mask], flops_arr[mask])
    if not res:
        raise ValueError("Could not compute a Pareto front from the provided rows")

    pareto_flops, pareto_losses, _pareto_indices = res
    fit_result = fit_power_law(
        np.asarray(pareto_flops, dtype=float), np.asarray(pareto_losses, dtype=float)
    )
    if fit_result is None:
        raise ValueError("Could not fit a power law to the loss from the provided rows")
    return fit_result


def predict_loss(
    fit: LogLogPowerLawFit, target_flop: float
) -> tuple[float, PowerLawLossInfo]:
    """The loss the power law predicts at `target_flop`."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive for a log-log power law")

    predicted_loss, predicted_loss_upper, predicted_loss_lower = predict_from_loglog_fit(
        fit=fit,
        target_flop=target_flop,
        z_score=1.96,
    )
    info = PowerLawLossInfo(
        slope=fit.slope,
        intercept=fit.intercept,
        r_squared=fit.r_squared,
        slope_std_err=fit.slope_std_err,
        num_pareto_points=fit.num_points,
        predicted_loss=predicted_loss,
        predicted_loss_upper=predicted_loss_upper,
        predicted_loss_lower=predicted_loss_lower,
    )
    return predicted_loss, info


# --- COARSE ----------------------------------------------------------------------


def prepare_rows_for_coarse_prediction(
    rows: list[Row] | list[list[Row]],
) -> tuple[list[Row], list[Row]]:
    """Rows to fit the model size on, and all rows.

    Plain rows keep the lowest loss per flops value; grouped learning-curve rows
    become their loss envelope.
    """
    all_rows = flatten_rows(rows)

    if rows and isinstance(rows[0], tuple):
        unique_flops_min_loss: dict[float, Row] = {}
        for row in all_rows:
            n_param, n_data, loss, flop = row
            if not (
                np.isfinite(n_param)
                and np.isfinite(n_data)
                and np.isfinite(loss)
                and np.isfinite(flop)
            ):
                continue
            if n_param <= 0 or n_data <= 0 or loss <= 0 or flop <= 0:
                continue

            existing = unique_flops_min_loss.get(float(flop))
            if existing is None or float(loss) < float(existing[2]):
                unique_flops_min_loss[float(flop)] = (
                    float(n_param),
                    float(n_data),
                    float(loss),
                    float(flop),
                )

        selected_rows = [
            unique_flops_min_loss[f] for f in sorted(unique_flops_min_loss.keys())
        ]
        return selected_rows, all_rows

    if not rows:
        return [], []

    return _learning_curve_envelope(cast("list[list[Row]]", rows)), all_rows


def _smoothed_curve(
    curve: list[Row], log_flops_grid: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    """One learning curve on the part of `log_flops_grid` it covers.

    Returns (n_params, grid mask, smoothed log10 loss, log10 n_data), or None if the
    curve has fewer than two usable points. Never extrapolates.
    """
    curve_arr = np.asarray(curve, dtype=float)
    if curve_arr.ndim != 2 or curve_arr.shape[1] != 4:
        return None

    n_params, n_data, losses, flops = curve_arr.T
    valid = (
        np.isfinite(n_params)
        & np.isfinite(losses)
        & np.isfinite(flops)
        & (n_params > 0)
        & (losses > 0)
        & (flops > 0)
    )
    n_params, n_data, losses, flops = (
        n_params[valid],
        n_data[valid],
        losses[valid],
        flops[valid],
    )

    # First occurrence per flops value (np.unique returns it), sorted by flops.
    flops, first = np.unique(flops, return_index=True)
    if len(flops) < 2:
        return None
    n_data, losses = n_data[first], losses[first]

    log_flops = np.log10(flops)
    covered = (log_flops_grid >= log_flops[0] - 1e-12) & (
        log_flops_grid <= log_flops[-1] + 1e-12
    )
    if not np.any(covered):
        return None

    log_loss = np.interp(log_flops_grid[covered], log_flops, np.log10(losses))
    if len(log_flops_grid) > 1:
        sigma = LC_SMOOTHING_SIGMA / (log_flops_grid[1] - log_flops_grid[0])
        log_loss = gaussian_filter1d(log_loss, sigma=sigma, mode="nearest")

    with np.errstate(divide="ignore", invalid="ignore"):
        log_data = np.interp(log_flops_grid[covered], log_flops, np.log10(n_data))
    return float(np.median(n_params)), covered, log_loss, log_data


def _learning_curve_envelope(curves: list[list[Row]]) -> list[Row]:
    """Chinchilla approach 1: the most efficient (n_params, n_data) at each budget.

    Every learning curve is interpolated and smoothed into a continuous map from
    flops to loss; at `LC_ENVELOPE_NUM_FLOPS` log-spaced flops values the run with
    the lowest loss gives that budget's (n_params, n_data, loss, flops) row.
    """
    curve_flops = [
        float(point[3]) for curve in curves for point in curve if len(point) == 4
    ]
    curve_flops = [f for f in curve_flops if np.isfinite(f) and f > 0]
    if not curve_flops or min(curve_flops) == max(curve_flops):
        return []
    log_flops_grid = np.linspace(
        np.log10(min(curve_flops)), np.log10(max(curve_flops)), LC_ENVELOPE_NUM_FLOPS
    )

    best_log_loss = np.full(len(log_flops_grid), np.inf)
    best_n_params = np.full(len(log_flops_grid), np.nan)
    best_log_data = np.full(len(log_flops_grid), np.nan)
    for curve in curves:
        smoothed = _smoothed_curve(curve, log_flops_grid)
        if smoothed is None:
            continue
        n_params, covered, log_loss, log_data = smoothed
        idx = np.flatnonzero(covered)
        better = log_loss < best_log_loss[idx]
        best_log_loss[idx[better]] = log_loss[better]
        best_n_params[idx[better]] = n_params
        best_log_data[idx[better]] = log_data[better]

    keep = (
        np.isfinite(best_log_loss)
        & np.isfinite(best_n_params)
        & np.isfinite(best_log_data)
    )
    return [
        (float(n), float(10.0**d), float(10.0**loss), float(10.0**f))
        for n, d, loss, f in zip(
            best_n_params[keep],
            best_log_data[keep],
            best_log_loss[keep],
            log_flops_grid[keep],
            strict=True,
        )
    ]


def fit_coarse(rows: list[Row] | list[list[Row]]) -> LogLogPowerLawFit:
    """Fit log10(n_params) against log10(flops) on the compute optimal rows."""
    rows_for_fit, _all_rows = prepare_rows_for_coarse_prediction(rows)

    rows_arr = np.asarray(rows_for_fit, dtype=float)
    if rows_arr.ndim != 2 or rows_arr.shape[1] != 4:
        raise ValueError("rows must be a 2D array-like with shape (n, 4)")

    params_arr = rows_arr[:, 0]
    data_arr = rows_arr[:, 1]
    flops_arr = rows_arr[:, 3]
    mask = (
        np.isfinite(params_arr)
        & np.isfinite(data_arr)
        & np.isfinite(flops_arr)
        & (params_arr > 0)
        & (data_arr > 0)
        & (flops_arr > 0)
    )
    if not np.any(mask):
        raise ValueError("rows do not contain any finite positive loss/flop values")

    fit_result = fit_power_law(
        np.asarray(flops_arr[mask], dtype=float),
        np.asarray(params_arr[mask], dtype=float),
    )
    if fit_result is None:
        raise ValueError(
            "Could not fit a power law to the model size from the provided rows"
        )
    return fit_result


def predict_coarse(
    fit: LogLogPowerLawFit, target_flop: float
) -> tuple[tuple[float, float], PowerLawCoarseInfo]:
    """The compute optimal `(n_params, n_data)` the power law predicts at a budget."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive for a log-log power law")

    predicted_param, _upper, _lower = predict_from_loglog_fit(
        fit=fit, target_flop=target_flop, z_score=1.96
    )
    info = PowerLawCoarseInfo(
        slope=fit.slope,
        intercept=fit.intercept,
        r_squared=fit.r_squared,
        slope_std_err=fit.slope_std_err,
    )
    predicted_n_param = float(predicted_param)
    if not np.isfinite(predicted_n_param) or predicted_n_param <= 0:
        raise ValueError("Could not predict a positive n_param from the power law")
    predicted_n_data = float(target_flop / (6.0 * predicted_n_param))
    if not np.isfinite(predicted_n_data) or predicted_n_data <= 0:
        raise ValueError("Could not derive a positive n_data from the predicted n_param")

    return (predicted_n_param, predicted_n_data), info


# --- FULL ------------------------------------------------------------------------


def compute_full_pareto(
    *,
    configs: list[dict[str, Any]],
    losses: list[float],
    flops: list[float],
) -> tuple[list[int], list[dict[str, Any]], list[float], list[float]]:
    """(ids, configs, flops, losses) of the (flops, loss) Pareto front."""
    pareto_result = pareto_front(losses, flops)
    if pareto_result is None:
        raise ValueError("Could not compute Pareto front from losses/flops")

    pareto_flops, pareto_losses, pareto_ids = pareto_result
    pareto_configs = [configs[i] for i in pareto_ids]
    return pareto_ids, pareto_configs, pareto_flops, pareto_losses


def _fits_in_log_space(param: Any) -> bool:
    """Whether an HP's power law is fit (and so extrapolated) in log-log space.

    Only the parameter's own declared log flag decides. `is_scaling` used to force
    it too, which put ordinal-index HPs (e.g. OpenCLIP's vision_width/text_width,
    indices into WIDTH_CHOICES) in log space, where a power law in the index is not
    a power law in the value.
    """
    return isinstance(param, HPOInteger | HPOFloat) and param.log


def fit_hp_power_laws(
    *,
    space: SearchSpace,
    pareto_configs: list[dict[str, Any]],
    pareto_flops: list[float],
) -> dict[str, LogLogPowerLawFit]:
    """Fit one power law per searchable HP, independent of any target_flops."""
    x_flops = np.asarray(pareto_flops, dtype=float)

    hp_fits: dict[str, LogLogPowerLawFit] = {}
    for param_name, param in space.searchables.items():
        hp_values_param: list[float] = []
        for config in pareto_configs:
            if param_name not in config:
                raise ValueError(f"HP '{param_name}' is missing from config: {config}")
            hp_values_param.append(float(config[param_name]))

        fit_result = fit_power_law(
            x_flops,
            np.asarray(hp_values_param, dtype=float),
            log_scale=_fits_in_log_space(param),
        )
        if fit_result is None:
            raise ValueError(
                f"Not enough valid data points to fit scaling law for HP '{param_name}'"
            )
        hp_fits[param_name] = fit_result

    return hp_fits


def fit_full(
    *,
    space: SearchSpace,
    configs: list[dict[str, Any]],
    losses: list[float],
    flops: list[float],
) -> PowerLawFullFit:
    """Fit one power law per searchable HP on the Pareto front of the configs."""
    pareto_ids, pareto_configs, pareto_flops, pareto_losses = compute_full_pareto(
        configs=configs, losses=losses, flops=flops
    )
    hp_fits = fit_hp_power_laws(
        space=space, pareto_configs=pareto_configs, pareto_flops=pareto_flops
    )
    return PowerLawFullFit(
        hp_fits=hp_fits,
        pareto_ids=pareto_ids,
        pareto_configs=pareto_configs,
        pareto_flops=pareto_flops,
        pareto_losses=pareto_losses,
    )


def _snap_to_log_base_grid(value: float, param: Any) -> float:
    """Round an integer log-base HP to the nearest power of its base.

    The optimizers only sample such HPs on that grid (e.g. 16, 32, 64 for base 2), but
    the log-log extrapolation lands in between (e.g. 20); round the exponent so the
    predicted config stays on the grid. Rounding is in log space, so 20 -> 16, 23 -> 32.
    """
    domain = param.domain
    if not (domain.is_int and domain.log_base is not None and value > 0):
        return value
    exponent = round(np.log(value) / np.log(domain.log_base))
    return float(np.clip(domain.log_base**exponent, domain.lower, domain.upper))


def predict_full(
    fit: PowerLawFullFit, space: SearchSpace, target_flops: float
) -> dict[str, Any]:
    """The config whose HPs the per-HP power laws extrapolate to `target_flops`."""
    extrapolated_config: dict[str, Any] = {}
    for param_name, param in space.searchables.items():
        predicted_hp, _, _ = predict_from_loglog_fit(
            fit.hp_fits[param_name],
            target_flops,
            log_scale=_fits_in_log_space(param),
        )
        predicted_hp = np.clip(predicted_hp, param.domain.lower, param.domain.upper)
        predicted_hp = _snap_to_log_base_grid(float(predicted_hp), param)
        extrapolated_config[param_name] = param.domain.cast_one(
            float(predicted_hp), frm=param.domain
        )

    return {
        "extrapolated_config": extrapolated_config,
        "pareto_ids": fit.pareto_ids,
        "pareto_configs": fit.pareto_configs,
        "pareto_flops": fit.pareto_flops,
        "pareto_losses": fit.pareto_losses,
    }


# --- Cache rows ------------------------------------------------------------------


def fit_to_row(fit: LogLogPowerLawFit | None) -> list[float]:
    """Serialize a fit to a fixed-width row; a failed fit becomes an all-NaN row."""
    if fit is None:
        return [float("nan")] * ROW_WIDTH
    return [
        fit.slope,
        fit.intercept,
        fit.r_squared,
        fit.slope_std_err,
        fit.intercept_std_err,
        float(fit.num_points),
    ]


def fit_from_row(row: np.ndarray) -> LogLogPowerLawFit | None:
    """Deserialize a fit row, None for a failed fit.

    Only a NaN slope/intercept means the fit failed: `linregress` legitimately returns
    NaN r_squared/stderr (division by zero) when the fitted quantity is constant across
    the Pareto front, even though slope/intercept are perfectly valid.
    """
    row = np.asarray(row, dtype=float)
    if np.any(np.isnan(row[:2])):
        return None
    slope, intercept, r_squared, slope_std_err, intercept_std_err, num_points = row
    return LogLogPowerLawFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        slope_std_err=float(slope_std_err),
        intercept_std_err=float(intercept_std_err),
        num_points=int(num_points) if np.isfinite(num_points) else 0,
    )
