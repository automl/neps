"""GP posterior prediction: a GP trained on the observed configurations, queried for
the best cost-feasible configuration at a compute budget.

One trained GP answers every target: `LOSS` is its posterior mean at that
configuration, `FULL` the configuration itself, and `COARSE` the configuration's
model size, with `n_data = flops / (6 * n_params)`.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GPPosteriorPredictionInfo:
    """Details of a GP posterior prediction at one compute budget."""

    num_trials_used: int
    target_flop: float
    predicted_loss: float
    constrained: bool  # True if a cost-feasible candidate was found


#: Version of the GP model, checkpoints of other versions are refit. Version 2 can
#: model `log(loss)`.
GP_FIT_VERSION = 2


def gp_cache_path(cache_dir: Path, n_trials: int) -> Path:
    """Checkpoint path of the GP fit on the first `n_trials` trials."""
    # Versioned, so neither the fits nor the prediction memos of others are reused
    return cache_dir / f"fit_gp_v{GP_FIT_VERSION}_n{n_trials}.pt"


def gp_pred_cache_path(cache_path: Path) -> Path:
    """Path of the prediction memo stored next to the fit checkpoint `cache_path`."""
    return cache_path.with_name(cache_path.stem + "_preds.pkl")


@dataclass
class GPPosteriorFit:
    """A target-flop-independent trained GP posterior over observed configs.

    Training (the expensive part) doesn't depend on target_flop at all. Call
    `.predict(target_flop)` to draw a fresh Sobol candidate pool, evaluate the
    already-trained posterior over it, and select the best cost-feasible
    candidate for that budget -- without retraining the GP.
    """

    space: Any
    encoder: Any
    gp: Any
    cost_estimator: Callable
    num_trials_used: int
    x_train: Any
    y_train: Any
    mean_slope: float
    mean_bias: float
    min_flops: int
    # Whether the GP models log(loss), its posterior mean is then exponentiated to the
    # median of the predicted loss
    log_y: bool = False
    # Sidecar file memoizing predict() results per target_flop (set by
    # fit_gp_posterior when a fit cache path is given). Predicting is the slow part
    # (Sobol pool + a Python cost_estimator loop), and is deterministic in
    # target_flop, so results are persisted next to the fit checkpoint.
    pred_cache_path: Path | None = None
    # Seed of the study being predicted (e.g. 40 for <optimizer>/seed_40). Drives the
    # Sobol pool and RNGs. Required at predict time.
    seed: int | None = None
    _pred_memo: dict | None = field(default=None, repr=False, compare=False)

    def _load_pred_memo(self) -> dict:
        if self._pred_memo is None:
            memo: dict = {}
            if self.pred_cache_path is not None and self.pred_cache_path.exists():
                try:
                    with self.pred_cache_path.open("rb") as f:
                        loaded = pickle.load(f)  # noqa: S301
                    if isinstance(loaded, dict):
                        memo = loaded
                except Exception:  # noqa: BLE001
                    memo = {}
            self._pred_memo = memo
        return self._pred_memo

    def _save_pred_memo(self) -> None:
        if self.pred_cache_path is None or self._pred_memo is None:
            return
        tmp = self.pred_cache_path.with_name(
            f"{self.pred_cache_path.name}.{os.getpid()}.tmp"
        )
        try:
            self.pred_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as f:
                pickle.dump(self._pred_memo, f)
            tmp.replace(self.pred_cache_path)  # atomic: safe with parallel array jobs
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Warning: could not save GP prediction cache to"
                f" {self.pred_cache_path}: {e}"
            )
            tmp.unlink(missing_ok=True)

    def predict(
        self,
        target_flop: float,
        margin: float = 0.1,
        num_candidates: int = 2048,
    ) -> tuple[dict[str, Any], float, GPPosteriorPredictionInfo]:
        """Cached wrapper around `_predict_uncached`, keyed by the call arguments."""
        key = (float(target_flop), float(margin), int(num_candidates), self.seed)
        memo = self._load_pred_memo()
        if key in memo:
            return memo[key]
        result = self._predict_uncached(target_flop, margin, num_candidates)
        memo[key] = result
        self._save_pred_memo()
        return result

    def _predict_uncached(  # noqa: C901
        self,
        target_flop: float,
        margin: float,
        num_candidates: int,
    ) -> tuple[dict[str, Any], float, GPPosteriorPredictionInfo]:
        """Draw Sobol candidates, evaluate the trained posterior over them, and
        pick the optimal config at or under target_flop.

        Candidates are drawn via Sobol sampling over the search space and filtered by
        estimated cost so the result never exceeds ``target_flop`` (matching the
        cost-constrained candidate construction used by ``IT-modified-JES`` rather than
        optimizing the GP posterior unconstrained and hoping a sample happens to land
        near ``target_flop``). The search first tries candidates within
        ``target_flop * [1 - margin, 1]`` and picks whichever minimizes predicted loss;
        if that band is empty it falls back to the closest-to-budget feasible candidate
        (``flops <= target_flop``); if no candidate is feasible at all, it falls back to
        the cheapest candidate found.
        """
        import random

        import torch
        from botorch.utils.sampling import draw_sobol_samples

        if target_flop <= 0:
            raise ValueError("target_flop must be positive")

        # Seed every RNG from the study seed (so <optimizer>/seed_40 always uses 40),
        # making predictions reproducible per study.
        if self.seed is None:
            raise ValueError("GP posterior prediction requires a study seed")
        seed = int(self.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        lower = [domain.lower for domain in self.encoder.domains]
        upper = [domain.upper for domain in self.encoder.domains]
        bounds = torch.tensor([lower, upper], dtype=torch.float64)

        candidate_set = (
            draw_sobol_samples(bounds=bounds, n=num_candidates, q=1, seed=seed)
            .squeeze(1)
            .to(dtype=torch.float64)
        )
        decoded_candidates = self.encoder.decode(candidate_set)

        flops = torch.full((len(decoded_candidates),), float("inf"), dtype=torch.float64)
        for i, config in enumerate(decoded_candidates):
            try:
                flops[i] = self.cost_estimator(**config)
            except Exception:  # noqa: BLE001, S112
                continue
        if torch.isinf(flops).all():
            raise ValueError("cost_estimator failed for every Sobol candidate")

        with torch.no_grad():
            predicted_loss = (
                self.gp.posterior(candidate_set.unsqueeze(-2))
                .mean.squeeze(-1)
                .squeeze(-1)
            )
        if self.log_y:
            predicted_loss = predicted_loss.exp()

        lo = target_flop * (1.0 - margin)

        def _best_in(mask: torch.Tensor) -> int | None:
            idxs = mask.nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                return None
            best_local = int(torch.argmin(predicted_loss[idxs]).item())
            return int(idxs[best_local].item())

        # 1. Prefer candidates in the target band, closest to budget: [lo, target_flop].
        best_idx = _best_in((flops >= lo) & (flops <= target_flop))
        # 2. Fall back to any feasible candidate (flops <= target_flop).
        if best_idx is None:
            best_idx = _best_in(flops <= target_flop)
        constrained = best_idx is not None

        # 3. No feasible candidate at all -- use the cheapest one found.
        if best_idx is None:
            logger.info(
                "WARNING: No Sobol candidate had estimated cost <= target_flop. "
                "Falling back to the cheapest candidate found."
            )
            best_idx = int(torch.argmin(flops).item())

        best_config = decoded_candidates[best_idx]
        best_output = float(predicted_loss[best_idx].item())

        info = GPPosteriorPredictionInfo(
            num_trials_used=self.num_trials_used,
            target_flop=target_flop,
            predicted_loss=best_output,
            constrained=constrained,
        )
        return best_config, best_output, info


#: The prior mean of the GP, the one of `BayesianOptimization.prior_mean` the observations
#: were acquired with.
PriorMean = Literal["power_law", "zero"]


def _model_info(*, log_y: bool, prior_mean: PriorMean = "power_law") -> dict[str, Any]:
    """What a checkpoint stores of the GP model, to only load matching ones."""
    info: dict[str, Any] = {"version": GP_FIT_VERSION, "log_y": log_y}
    # Only when not the default, so the checkpoints written before it still load
    if prior_mean != "power_law":
        info["prior_mean"] = prior_mean
    return info


def _train_gp(
    *,
    space: Any,
    configs: list[dict[str, Any]],
    losses: list[float],
    cost_estimator: Callable,
    log_y: bool = True,
    prior_mean: PriorMean = "power_law",
) -> GPPosteriorFit:
    """Train a GP on observed configs/losses. Independent of any target_flop.

    The GP is the one `BayesianOptimization` builds with `prior_mean`, see
    `make_single_obj_gp()`.
    """
    import torch
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from scalefm.optimizers.models.gp import make_single_obj_gp
    from scalefm.space import ConfigEncoder
    from scalefm.utils.power_law import fit_power_law_mean_params

    if cost_estimator is None:
        raise ValueError("cost_estimator is required for GP posterior prediction")

    parameters = space.searchables
    encoder = ConfigEncoder.from_parameters(parameters)

    valid = [
        (c, float(loss))
        for c, loss in zip(configs, losses, strict=False)
        if np.isfinite(float(loss))
    ]
    if len(valid) < 2:
        raise ValueError(f"Need at least 2 finite loss values, got {len(valid)}")

    valid_configs, valid_losses = zip(*valid, strict=False)

    x_train = encoder.encode(list(valid_configs))
    y_train = torch.tensor(valid_losses, dtype=torch.float64).unsqueeze(-1)

    # Fit a power law to determine the GP mean prior, matching the scaling-law
    # kernel used by BayesianOptimization.sample_candidates. Unused by the zero mean.
    mean_params = fit_power_law_mean_params(valid_configs, valid_losses, cost_estimator)

    gp = make_single_obj_gp(
        x=x_train,
        y=y_train,
        encoder=encoder,
        prior_mean=prior_mean,
        cost_estimator=cost_estimator,
        min_cost=mean_params.min_flops,
        mean_slope=mean_params.slope,
        mean_bias=mean_params.intercept,
        log_y=log_y,
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()

    return GPPosteriorFit(
        space=space,
        encoder=encoder,
        gp=gp,
        cost_estimator=cost_estimator,
        num_trials_used=len(valid),
        x_train=x_train,
        y_train=y_train,
        mean_slope=mean_params.slope,
        mean_bias=mean_params.intercept,
        min_flops=mean_params.min_flops,
        log_y=gp.log_y,
    )


def _save_gp_fit(
    path: Path, n_trials: int, fit: GPPosteriorFit, model_info: dict[str, Any]
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(
            {
                "n_trials": n_trials,
                "gp_state_dict": fit.gp.state_dict(),
                "x_train": fit.x_train,
                "y_train": fit.y_train,
                "mean_slope": fit.mean_slope,
                "mean_bias": fit.mean_bias,
                "min_flops": fit.min_flops,
                "model_info": model_info,
            },
            path,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Warning: could not save GP fit cache to {path}: {e}")


def _load_gp_fit(
    path: Path,
    n_trials: int,
    *,
    space: Any,
    cost_estimator: Callable,
    log_y: bool = True,
    prior_mean: PriorMean = "power_law",
) -> GPPosteriorFit | None:
    import torch

    from scalefm.optimizers.models.gp import make_single_obj_gp
    from scalefm.space import ConfigEncoder

    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, weights_only=True)
        if int(checkpoint["n_trials"]) != n_trials:
            return None
        if checkpoint.get("model_info") != _model_info(
            log_y=log_y, prior_mean=prior_mean
        ):
            return None

        encoder = ConfigEncoder.from_parameters(space.searchables)
        x_train = checkpoint["x_train"]
        y_train = checkpoint["y_train"]
        gp = make_single_obj_gp(
            x=x_train,
            y=y_train,
            encoder=encoder,
            prior_mean=prior_mean,
            cost_estimator=cost_estimator,
            min_cost=checkpoint["min_flops"],
            mean_slope=checkpoint["mean_slope"],
            mean_bias=checkpoint["mean_bias"],
            log_y=log_y,
        )
        gp.load_state_dict(checkpoint["gp_state_dict"])
        gp.eval()

        return GPPosteriorFit(
            space=space,
            encoder=encoder,
            gp=gp,
            cost_estimator=cost_estimator,
            num_trials_used=n_trials,
            x_train=x_train,
            y_train=y_train,
            mean_slope=checkpoint["mean_slope"],
            mean_bias=checkpoint["mean_bias"],
            min_flops=checkpoint["min_flops"],
            log_y=gp.log_y,
            pred_cache_path=gp_pred_cache_path(path),
        )
    except Exception:  # noqa: BLE001
        return None


def fit_gp_posterior(
    *,
    space: Any,
    configs: list[dict[str, Any]],
    losses: list[float],
    cost_estimator: Callable,
    cache_path: Path | None = None,
    override: bool = False,
    seed: int | None = None,
    log_y: bool = True,
    prior_mean: PriorMean = "power_law",
) -> GPPosteriorFit:
    """Fit (or load from cache) a GP posterior over `configs`/`losses`.

    `log_y` is handed to `make_default_single_obj_gp()`, see there. `prior_mean` is the
    one the optimizer acquired the observations with, see `make_single_obj_gp()`.

    Caching, when `cache_path` is given, is keyed by `n_trials = len(configs)`
    only -- never by target_flop, since training doesn't depend on it. Call
    `.predict(target_flop)` on the result to acquire a point for any target
    without retraining. Pass `override=True` to ignore any cached fit and
    retrain from scratch, overwriting the cache file with the new result.
    """
    if seed is None:
        raise ValueError(
            "fit_gp_posterior requires a study seed (e.g. 40 for <optimizer>/seed_40)"
        )
    n_trials = len(configs)

    if cache_path is not None and not override:
        cached = _load_gp_fit(
            cache_path,
            n_trials,
            space=space,
            cost_estimator=cost_estimator,
            log_y=log_y,
            prior_mean=prior_mean,
        )
        if cached is not None:
            cached.seed = seed
            return cached

    fit = _train_gp(
        space=space,
        configs=configs,
        losses=losses,
        cost_estimator=cost_estimator,
        log_y=log_y,
        prior_mean=prior_mean,
    )

    fit.seed = seed
    if cache_path is not None:
        _save_gp_fit(
            cache_path, n_trials, fit, _model_info(log_y=log_y, prior_mean=prior_mean)
        )
        # A memo of an earlier fit at this path would no longer match the new fit
        gp_pred_cache_path(cache_path).unlink(missing_ok=True)

    return fit


def fit(
    *,
    space: Any,
    configs: list[dict[str, Any]],
    losses: list[float],
    cost_estimator: Callable,
    cache_dir: Path | None = None,
    seed: int | None = None,
    override: bool = False,
    log_y: bool = True,
    prior_mean: PriorMean = "power_law",
) -> GPPosteriorFit:
    """`fit_gp_posterior` with its checkpoint at `gp_cache_path(cache_dir, len(configs))`.

    The checkpoint only depends on the trial prefix, so every target shares it.
    """
    cache_path = gp_cache_path(cache_dir, len(configs)) if cache_dir is not None else None
    return fit_gp_posterior(
        space=space,
        configs=configs,
        losses=losses,
        cost_estimator=cost_estimator,
        cache_path=cache_path,
        seed=seed,
        override=override,
        log_y=log_y,
        prior_mean=prior_mean,
    )


def predict_loss(
    fit: GPPosteriorFit, target_flop: float
) -> tuple[float, GPPosteriorPredictionInfo]:
    """The GP posterior mean loss at the best config for `target_flop`."""
    _best_config, predicted_loss, info = fit.predict(target_flop)
    return predicted_loss, info


def predict_coarse(
    fit: GPPosteriorFit, params_estimator: Callable, target_flop: float
) -> tuple[tuple[float, float], GPPosteriorPredictionInfo]:
    """The `(n_params, n_data)` of the best config for `target_flop`."""
    best_config, _, info = fit.predict(target_flop)
    n_param = float(params_estimator(**best_config))
    if not np.isfinite(n_param) or n_param <= 0:
        raise ValueError(
            "params_estimator returned a non-positive value for the GP-predicted config"
        )
    n_data = float(target_flop / (6.0 * n_param))
    if not np.isfinite(n_data) or n_data <= 0:
        raise ValueError("Could not derive a positive n_data from GP-predicted n_param")
    return (n_param, n_data), info


def predict_full(fit: GPPosteriorFit, target_flops: float) -> dict[str, Any]:
    """The best config for `target_flops`, with its predicted loss."""
    best_config, predicted_loss, info = fit.predict(target_flops)
    return {
        "extrapolated_config": best_config,
        "predicted_loss": predicted_loss,
        "gp_posterior_info": info,
        "pareto_ids": [],
        "pareto_configs": [],
        "pareto_flops": [],
        "pareto_losses": [],
    }
