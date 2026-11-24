from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scalefm
from scalefm.prediction import (
    PredictorTypes,
    Row,
    ScalingLawParameters,
    Target,
    chinchilla3,
    fit_and_predict,
    gp,
    power_law,
)
from scalefm.prediction.scaling_law_fit import fit_power_law, predict_from_loglog_fit
from scalefm.space import HPOFloat, HPOInteger, SearchSpace

TRUE_LAW = ScalingLawParameters(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)

SPACE = SearchSpace(
    {
        "n_params": HPOFloat(1e6, 1e9, log=True),
        "n_data": HPOFloat(1e8, 1e11, log=True),
    }
)


def _flops(n_params: float, n_data: float) -> float:
    return 6.0 * n_params * n_data


def _synthetic(n: int = 300, seed: int = 0) -> list[Row]:
    rng = np.random.default_rng(seed)
    n_params = 10 ** rng.uniform(6, 9, n)
    n_data = 10 ** rng.uniform(8, 11, n)
    losses = (
        TRUE_LAW.E
        + TRUE_LAW.A / n_params**TRUE_LAW.alpha
        + TRUE_LAW.B / n_data**TRUE_LAW.beta
    )
    return [
        (float(p), float(d), float(loss), _flops(p, d))
        for p, d, loss in zip(n_params, n_data, losses, strict=True)
    ]


def _configs(rows: list[Row]) -> list[dict[str, float]]:
    return [{"n_params": r[0], "n_data": r[1]} for r in rows]


def test_predict_from_loglog_fit_round_trips_and_orders_bounds() -> None:
    flops = np.array([1e15, 1e16, 1e17, 1e18])
    fit = fit_power_law(flops, 10.0 * flops**-0.05 * [1.0, 1.01, 0.99, 1.0])
    assert fit is not None
    mean, upper, lower = predict_from_loglog_fit(fit, 1e20)
    assert lower < mean < upper
    with pytest.raises(ValueError):
        predict_from_loglog_fit(fit, 0.0)


@pytest.mark.parametrize("target_flops", [1e19, 1e21, 1e23])
def test_chinchilla3_recovers_the_compute_optimum(target_flops: float) -> None:
    fit = chinchilla3.fit_coarse(_synthetic(), best_loss_guess=TRUE_LAW.E * 1.5)
    n_params, n_data, loss = chinchilla3.optimum(TRUE_LAW, target_flops)

    (pred_n_params, pred_n_data), _info = chinchilla3.predict_coarse(fit, target_flops)
    pred_loss, _params = chinchilla3.predict_loss(fit, target_flops)
    assert pred_loss == pytest.approx(loss, rel=1e-3)
    assert pred_n_params == pytest.approx(n_params, rel=1e-2)
    assert pred_n_data == pytest.approx(n_data, rel=1e-2)

    row = chinchilla3.fit_to_row(fit)
    assert len(row) == chinchilla3.ROW_WIDTH
    assert chinchilla3.fit_from_row(np.asarray(row)) == fit


def test_power_law_recovers_a_pure_power_law() -> None:
    flops = np.logspace(15, 19, 12)
    rows: list[Row] = [(1.0, 1.0, 5.0 * c**-0.05, float(c)) for c in flops]
    # Off the front, so ignored
    rows.append((1.0, 1.0, 100.0, 1e18))

    fit = power_law.fit_loss(rows)
    loss, info = power_law.predict_loss(fit, 1e22)

    assert loss == pytest.approx(5.0 * 1e22**-0.05)
    assert info.predicted_loss_lower <= loss <= info.predicted_loss_upper
    assert power_law.fit_from_row(np.asarray(power_law.fit_to_row(fit))) == fit


def test_power_law_coarse_on_rows_and_learning_curves() -> None:
    # n_params = 0.1 * flops^0.5 along the compute optimal frontier
    flops = np.logspace(15, 19, 10)
    rows: list[Row] = [
        (0.1 * c**0.5, c / (6 * 0.1 * c**0.5), 10.0 * c**-0.05, float(c)) for c in flops
    ]
    (n_params, n_data), _ = power_law.predict_coarse(power_law.fit_coarse(rows), 1e21)
    assert n_params == pytest.approx(0.1 * 1e21**0.5, rel=1e-6)
    assert n_data == pytest.approx(1e21 / (6 * n_params))

    curves = [
        [(n, f / (6 * n), 10.0 * f**-0.05 * (1 + k), f) for f in np.logspace(14, 18, 5)]
        for k, n in enumerate([1e6, 1e7])
    ]
    (n_params, _), _ = power_law.predict_coarse(power_law.fit_coarse(curves), 1e20)
    assert np.isfinite(n_params)
    assert n_params > 0


def test_power_law_full_clips_into_the_space() -> None:
    rows = _synthetic(n=80)
    fit = power_law.fit_full(
        space=SPACE,
        configs=_configs(rows),
        losses=[r[2] for r in rows],
        flops=[r[3] for r in rows],
    )
    result = power_law.predict_full(fit, SPACE, 1e30)
    config = result["extrapolated_config"]
    assert set(config) == {"n_params", "n_data"}
    assert 1e6 <= config["n_params"] <= 1e9
    assert 1e8 <= config["n_data"] <= 1e11


def test_power_law_full_fits_linear_hps_in_linear_space() -> None:
    """A linear HP keeps its zero values and is extrapolated linearly, not via 10**."""
    space = SearchSpace(
        {
            "n_params": HPOFloat(1e6, 1e9, log=True),
            "width": HPOInteger(0, 5),  # ordinal index, 0 is a valid value
            "beta1": HPOFloat(0.8, 0.99),
        }
    )
    flops = [1e15, 1e16, 1e17, 1e18]
    configs = [
        {"n_params": 1e6 * 10**i, "width": w, "beta1": 0.9}
        for i, w in enumerate([0, 0, 1, 2])
    ]
    fit = power_law.fit_full(
        space=space, configs=configs, losses=[4.0, 3.0, 2.0, 1.0], flops=flops
    )
    assert fit.hp_fits["width"].num_points == 4

    config = power_law.predict_full(fit, space, 1e18)["extrapolated_config"]
    assert config["beta1"] == pytest.approx(0.9)
    assert 0 <= config["width"] <= 5


def test_gp_posterior_predicts_a_feasible_config(tmp_path: Path) -> None:
    rows = _synthetic(n=20)
    fit = gp.fit(
        space=SPACE,
        configs=_configs(rows),
        losses=[r[2] for r in rows],
        cost_estimator=_flops,
        cache_dir=tmp_path,
        seed=1,
    )
    target = 1e17
    (n_params, n_data), info = gp.predict_coarse(
        fit, lambda n_params, **_: n_params, target
    )
    assert info.constrained
    assert _flops(n_params, n_data) == pytest.approx(target)
    full = gp.predict_full(fit, target)
    assert _flops(**full["extrapolated_config"]) <= target

    # The checkpoint loads back without retraining
    assert gp.gp_cache_path(tmp_path, 20).exists()
    reloaded = gp.fit(
        space=SPACE,
        configs=_configs(rows),
        losses=[r[2] for r in rows],
        cost_estimator=_flops,
        cache_dir=tmp_path,
        seed=1,
    )
    assert reloaded.predict(target)[1] == pytest.approx(fit.predict(target)[1])


def test_fit_and_predict_rejects_unprovided_targets() -> None:
    rows = _synthetic(n=20)
    with pytest.raises(ValueError, match="does not predict"):
        fit_and_predict(
            PredictorTypes.CHINCHILLA3,
            [1e21],
            targets=[Target.FULL],
            rows=rows,
            configs=_configs(rows),
        )


def test_chinchilla3_needs_three_rows() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        chinchilla3.fit_coarse([(1.0, 1.0, 1.0, 6.0)], best_loss_guess=1.0)


def _import_run(root: Path, n: int = 60, *, report_scale: bool = False) -> None:
    """A run of the synthetic law, whose reports have `n_params` and `n_data` with
    `report_scale`.
    """
    scalefm.import_trials(
        [
            (
                {"n_params": r[0], "n_data": r[1]},
                {"objective_to_minimize": r[2], "cost": r[3]}
                | ({"n_params": r[0], "n_data": r[1]} if report_scale else {}),
            )
            for r in _synthetic(n=n)
        ],
        root_directory=root,
        pipeline_space=SPACE,
        optimizer="random_search",
    )


def test_predict_from_a_run(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _import_run(root)

    # "auto" is the power law. Without a model size, COARSE is left out
    (power,) = scalefm.predict(root, 1e21)
    assert power.loss is not None
    assert power.loss_interval is not None
    assert power.config is not None
    assert power.n_params is None
    assert set(power.info or {}) == {Target.LOSS, Target.FULL}

    # Chinchilla's law, and the COARSE power law, are fitted on the reported scale
    with pytest.raises(ValueError, match="no `n_params` or `n_data`"):
        scalefm.predict(root, 1e21, predictor="chinchilla3")
    with pytest.raises(ValueError, match="no `n_params` or `n_data`"):
        scalefm.predict(root, 1e21, targets=[Target.COARSE])

    scaled = tmp_path / "scaled"
    _import_run(scaled, report_scale=True)
    (power,) = scalefm.predict(scaled, 1e21)
    assert power.n_params is not None
    predictions = scalefm.predict(scaled, [1e20, 1e21], predictor="chinchilla3")
    assert [p.target_cost for p in predictions] == [1e20, 1e21]
    n_params, _n_data, _loss = chinchilla3.optimum(TRUE_LAW, 1e21)
    assert predictions[1].n_params == pytest.approx(n_params, rel=0.05)

    with pytest.raises(ValueError, match="does not predict"):
        scalefm.predict(scaled, 1e21, predictor="chinchilla3", targets=[Target.FULL])
    with pytest.raises(ValueError, match="Unknown predictor"):
        scalefm.predict(root, 1e21, predictor="nope")


def test_predict_gp_posterior_from_a_run(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _import_run(root, n=15)

    (prediction,) = scalefm.predict(
        root,
        1e17,
        predictor="gp_posterior",
        targets=[Target.LOSS, Target.COARSE, Target.FULL],
        params_estimator=lambda n_params: n_params,
        cost_estimator=_flops,
    )
    assert prediction.loss is not None
    assert prediction.n_params is not None
    assert prediction.config is not None
    # One GP answers every target
    assert prediction.config["n_params"] == pytest.approx(prediction.n_params)


def test_gp_posterior_checkpoint_only_loads_with_the_same_log_y(
    tmp_path: Path,
) -> None:
    rows = _synthetic(n=12)
    gp.fit(
        space=SPACE,
        configs=_configs(rows),
        losses=[r[2] for r in rows],
        cost_estimator=_flops,
        cache_dir=tmp_path,
        seed=1,
    )
    path = gp.gp_cache_path(tmp_path, 12)
    kwargs = {"space": SPACE, "cost_estimator": _flops}
    assert gp._load_gp_fit(path, 12, **kwargs) is not None
    assert gp._load_gp_fit(path, 12, log_y=False, **kwargs) is None


def test_gp_posterior_with_the_zero_prior_mean_of_the_optimizer(tmp_path: Path) -> None:
    from gpytorch.means import ZeroMean

    from scalefm.optimizers.models.gp import PowerLawMean

    rows = _synthetic(n=12)
    kwargs = {
        "space": SPACE,
        "configs": _configs(rows),
        "losses": [r[2] for r in rows],
        "cost_estimator": _flops,
        "cache_dir": tmp_path,
        "seed": 1,
    }
    fit = gp.fit(**kwargs, prior_mean="zero")
    assert isinstance(fit.gp.mean_module, ZeroMean)

    # The checkpoint of one prior mean never loads as the other
    path = gp.gp_cache_path(tmp_path, 12)
    load_kwargs = {"space": SPACE, "cost_estimator": _flops}
    assert gp._load_gp_fit(path, 12, **load_kwargs) is None
    reloaded = gp._load_gp_fit(path, 12, prior_mean="zero", **load_kwargs)
    assert reloaded is not None
    assert isinstance(reloaded.gp.mean_module, ZeroMean)
    target = 1e17
    reloaded.seed = 1
    assert reloaded.predict(target)[1] == pytest.approx(fit.predict(target)[1])

    # Refitting with the power law replaces it
    power_law_fit = gp.fit(**kwargs)
    assert isinstance(power_law_fit.gp.mean_module, PowerLawMean)
    assert gp._load_gp_fit(path, 12, prior_mean="zero", **load_kwargs) is None
