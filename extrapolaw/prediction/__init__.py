"""Extrapolate the evaluated runs of a scaling study to a larger compute budget.

The optimizers decide which runs to evaluate, the predictors here fit a scaling law on
them and predict, at a target budget, the best loss (`LOSS`), the compute optimal model
size and data (`COARSE`), or the compute optimal configuration (`FULL`). Each predictor
lives in its own module (`power_law`, `chinchilla3`, `gp`) as a pair of `fit_*` and
`predict_*` functions per target, see [`extrapolaw.predict()`][extrapolaw.predict] for the
high level entry point.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from extrapolaw.prediction import chinchilla3, gp, power_law
from extrapolaw.prediction.base import (
    PredictorTypes,
    Row,
    Target,
    check_provides,
    flatten_rows,
    last_point_in_curve,
)
from extrapolaw.prediction.scaling_law_fit import (
    DEFAULT_EXPONENT_BOUNDS,
    ScalingLawParameters,
)

if TYPE_CHECKING:
    from pathlib import Path

    from extrapolaw.space import SearchSpace
    from extrapolaw.state import Trial


@dataclass
class Prediction:
    """The prediction at `target_flops`, the targets that were not asked are `None`."""

    target_flops: float
    loss: float | None = None
    loss_interval: tuple[float, float] | None = None
    """The `(lower, upper)` bounds of `loss`, if the predictor has them."""
    n_params: float | None = None
    n_data: float | None = None
    config: dict[str, Any] | None = None
    info: dict[Target, Any] | None = None
    """Predictor specific details per target, e.g. the fitted coefficients."""


def get_predictor_type(predictor: str | PredictorTypes) -> PredictorTypes:
    """A predictor type, or the one with that slug, e.g. `"power_law"`."""
    if isinstance(predictor, PredictorTypes):
        return predictor
    for predictor_type in PredictorTypes:
        if predictor_type.value == predictor:
            return predictor_type
    raise ValueError(
        f"Unknown predictor '{predictor}', choose one of"
        f" {[p.value for p in PredictorTypes]}."
    )


def rows_from_trials(
    trials: Iterable[Trial],
    *,
    params_estimator: Callable[..., float] | None = None,
    data_parameter_name: str | None = None,
    cost_estimator: Callable[..., float] | None = None,
) -> tuple[list[Row], list[list[Row]], list[dict[str, Any]]]:
    """The successfully evaluated trials, in the order they were sampled.

    Args:
        trials: The trials of a run.
        params_estimator: Called as `params_estimator(**config)` for `n_params`,
            `extra["n_param"]` of the report is used without it.
        data_parameter_name: The parameter holding `n_data`, `extra["n_data"]` of the
            report is used without it.
        cost_estimator: Called as `cost_estimator(**config)` for the flops of trials
            that reported no cost.

    Returns:
        `(rows, learning_curve_rows, configs)`: one `(n_params, n_data, loss, flops)`
        row and config per trial, and the learning curves, reported as rows of
        `(n_data, flops, loss, ...)`, of the trials that have at least two points.
    """
    rows: list[Row] = []
    curves: list[list[Row]] = []
    configs: list[dict[str, Any]] = []
    for trial in sorted(trials, key=lambda t: t.metadata.time_sampled):
        report = trial.report
        if report is None or report.err is not None or report.reported_as != "success":
            continue
        loss = report.objective_to_minimize
        loss = float("nan") if loss is None or isinstance(loss, Sequence) else loss
        loss = _nan_float(loss)
        if np.isinf(loss):
            loss = float("nan")

        config = dict(trial.config)
        extra = report.extra or {}
        flops = report.cost
        if flops is None and cost_estimator is not None:
            flops = cost_estimator(**config)
        n_params = (
            params_estimator(**config)
            if params_estimator is not None
            else extra.get("n_param")
        )
        n_data = (
            config.get(data_parameter_name)
            if data_parameter_name is not None
            else extra.get("n_data")
        )
        # The values the run reported come first
        if report.n_params is not None:
            n_params = report.n_params
        if report.n_data is not None:
            n_data = report.n_data
        n_params = _nan_float(n_params)
        rows.append((n_params, _nan_float(n_data), loss, _nan_float(flops)))
        configs.append(config)

        curve = [
            (n_params, _nan_float(point[0]), _nan_float(point[2]), _nan_float(point[1]))
            for point in report.learning_curve or []
            if isinstance(point, Sequence) and len(point) >= 3
        ]
        if len(curve) >= 2:
            curves.append(curve)
    return rows, curves, configs


def fit_and_predict(  # noqa: PLR0913
    predictor_type: PredictorTypes,
    target_flops: Sequence[float],
    *,
    targets: Iterable[Target],
    rows: list[Row],
    configs: list[dict[str, Any]],
    learning_curve_rows: list[list[Row]] | None = None,
    space: SearchSpace | None = None,
    params_estimator: Callable[..., float] | None = None,
    cost_estimator: Callable[..., float] | None = None,
    best_loss_guess: float | None = None,
    exponent_bounds: tuple[float, float] | None = None,
    seed: int = 0,
    cache_dir: Path | None = None,
    prior_mean: gp.PriorMean = "power_law",
) -> list[Prediction]:
    """Fit `predictor_type` once per target on all the rows and predict at every budget.

    The coarse POWER_LAW fit uses `learning_curve_rows` when there are any. GP_POSTERIOR
    fits a single GP that answers every target, and needs `space` and `cost_estimator`.
    Its prior mean is `prior_mean`, the one the optimizer acquired the rows with.
    """
    targets = list(dict.fromkeys(targets))
    for target in targets:
        check_provides(predictor_type, target)
    if not rows:
        raise ValueError("No successfully evaluated trials to fit on.")
    if best_loss_guess is None:
        best_loss_guess = float(np.nanmin(np.asarray([r[2] for r in rows], dtype=float)))

    needs_space = predictor_type == PredictorTypes.GP_POSTERIOR or Target.FULL in targets
    if needs_space and space is None:
        raise ValueError(f"{predictor_type.display_label} needs the search space.")

    fits = _fit(
        predictor_type,
        targets,
        rows=rows,
        configs=configs,
        learning_curve_rows=learning_curve_rows,
        space=space,
        cost_estimator=cost_estimator,
        best_loss_guess=best_loss_guess,
        exponent_bounds=exponent_bounds,
        seed=seed,
        cache_dir=cache_dir,
        prior_mean=prior_mean,
    )
    predictions = []
    for flops in target_flops:
        prediction = Prediction(target_flops=float(flops), info={})
        for target, fit in fits.items():
            _fill(prediction, predictor_type, target, fit, space, params_estimator)
        predictions.append(prediction)
    return predictions


def _fit(  # noqa: PLR0913
    predictor_type: PredictorTypes,
    targets: list[Target],
    *,
    rows: list[Row],
    configs: list[dict[str, Any]],
    learning_curve_rows: list[list[Row]] | None,
    space: SearchSpace | None,
    cost_estimator: Callable[..., float] | None,
    best_loss_guess: float,
    exponent_bounds: tuple[float, float] | None,
    seed: int,
    cache_dir: Path | None,
    prior_mean: gp.PriorMean = "power_law",
) -> dict[Target, Any]:
    """One fit per target, a GP_POSTERIOR fit is shared by every target."""
    losses = [float(r[2]) for r in rows]
    if predictor_type == PredictorTypes.GP_POSTERIOR:
        if cost_estimator is None:
            raise ValueError("GP_POSTERIOR needs a `cost_estimator`.")
        gp_fit = gp.fit(
            space=space,
            configs=configs,
            losses=losses,
            cost_estimator=cost_estimator,
            cache_dir=cache_dir,
            seed=seed,
            prior_mean=prior_mean,
        )
        return dict.fromkeys(targets, gp_fit)

    fits: dict[Target, Any] = {}
    for target in targets:
        if target == Target.FULL:
            fits[target] = power_law.fit_full(
                space=space,  # type: ignore[arg-type]
                configs=configs,
                losses=losses,
                flops=[float(r[3]) for r in rows],
            )
        elif predictor_type == PredictorTypes.POWER_LAW and target == Target.LOSS:
            fits[target] = power_law.fit_loss(rows)
        elif predictor_type == PredictorTypes.POWER_LAW:
            fits[target] = power_law.fit_coarse(learning_curve_rows or rows)
        elif target == Target.LOSS:
            fits[target] = chinchilla3.fit_loss(
                rows, best_loss_guess=best_loss_guess, exponent_bounds=exponent_bounds
            )
        else:
            fits[target] = chinchilla3.fit_coarse(
                rows, best_loss_guess=best_loss_guess, exponent_bounds=exponent_bounds
            )
    return fits


def _fill(
    prediction: Prediction,
    predictor_type: PredictorTypes,
    target: Target,
    fit: Any,
    space: SearchSpace | None,
    params_estimator: Callable[..., float] | None,
) -> None:
    """Predict `target` with `fit` at the budget of `prediction` and store it there."""
    flops = prediction.target_flops
    assert prediction.info is not None
    is_gp = predictor_type == PredictorTypes.GP_POSTERIOR
    module: Any = {
        PredictorTypes.POWER_LAW: power_law,
        PredictorTypes.CHINCHILLA3: chinchilla3,
        PredictorTypes.GP_POSTERIOR: gp,
    }[predictor_type]

    info: Any
    if target == Target.LOSS:
        prediction.loss, info = module.predict_loss(fit, flops)
        if isinstance(info, power_law.PowerLawLossInfo):
            bounds = (info.predicted_loss_lower, info.predicted_loss_upper)
            prediction.loss_interval = (min(bounds), max(bounds))
    elif target == Target.COARSE and is_gp:
        if params_estimator is None:
            raise ValueError("GP_POSTERIOR coarse prediction needs a params_estimator.")
        (prediction.n_params, prediction.n_data), info = gp.predict_coarse(
            fit, params_estimator, flops
        )
    elif target == Target.COARSE:
        (prediction.n_params, prediction.n_data), info = module.predict_coarse(fit, flops)
    else:
        info = (
            gp.predict_full(fit, flops)
            if is_gp
            else power_law.predict_full(fit, space, flops)  # type: ignore[arg-type]
        )
        prediction.config = dict(info["extrapolated_config"])
    prediction.info[target] = info


def _nan_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


__all__ = [
    "DEFAULT_EXPONENT_BOUNDS",
    "Prediction",
    "PredictorTypes",
    "Row",
    "ScalingLawParameters",
    "Target",
    "check_provides",
    "chinchilla3",
    "fit_and_predict",
    "flatten_rows",
    "get_predictor_type",
    "gp",
    "last_point_in_curve",
    "power_law",
    "rows_from_trials",
]
