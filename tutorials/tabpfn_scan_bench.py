"""TabPFN on the surrogate benchmark of `scan_bench`, for the tutorial.

A query predicts what a TabPFN training run would reach, so a run takes a fraction of
a second. The `evaluate_pipeline` of the tutorial is defined in the notebook, from
the pieces here: the surrogates, the `TabPFNConfig` of a config of the search space,
the parameter and FLOP estimators, and the surrogate's own best config at a FLOP
budget, to score predictions against.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

import numpy as np
from scan_bench import (
    PerformancePredictorType,
    TabPFNBenchmark,
    TabPFNConfig,
    TabPFNTarget,
)

# They define the prior construction (synthetic data pre-trained on), not its capacity, so they are not in the search space.
MAX_FEATURES = 32
NUM_DATAPOINTS_MAX = 128

# `TabPFNConfig` rejects fewer cells than this
MIN_CELLS = 2**20

DOWNSTREAM = {"real_nll": TabPFNTarget.NLL}
"""The downstream performances, by name."""

SIMULATED_FLOPS_PER_SECOND = 2e13
"""The speed of the cluster the tutorial simulates, to give runs a training time."""


@cache
def get_benchmark(
    target: TabPFNTarget = TabPFNTarget.VAL_LOSS,
    predictor_type: PerformancePredictorType = PerformancePredictorType.ENSEMBLE_XGB,
) -> TabPFNBenchmark:
    """The surrogate of `target`, built once per process. Only the validation loss
    ships fitted, the first call for another target fits it (about 20 seconds).
    """
    return TabPFNBenchmark(
        target=target.value, predictor_type=predictor_type, device="cpu"
    )


def build_config(
    lr: float,
    effective_batch_size: float,
    embedding_size: float,
    total_cells: float,
    num_layers: float,
    **_: float,
) -> TabPFNConfig:
    """The `TabPFNConfig` of a config of the search space."""
    return TabPFNConfig(
        total_cells=int(total_cells),
        effective_batch_size=int(effective_batch_size),
        lr=float(lr),
        max_features=MAX_FEATURES,
        embedding_size=int(embedding_size),
        num_layers=int(num_layers),
        num_datapoints_max=NUM_DATAPOINTS_MAX,
    )


@cache
def _json_table(name: str) -> list[dict]:
    """A lookup table shipped with `scan_bench.tabpfn`, read once per process."""
    with files("scan_bench.tabpfn").joinpath(name).open(encoding="utf-8") as f:
        return json.load(f)


@cache
def _n_params(embedding_size: int, num_layers: int) -> int:
    for entry in _json_table("n_params.json"):
        model = entry["model_config"]
        if (
            model["embedding_size"] == embedding_size
            and model["num_layers"] == num_layers
        ):
            return entry["n_parameters"]
    raise ValueError(f"No parameter count for {embedding_size=}, {num_layers=}")


@cache
def _flops_per_cell(embedding_size: int, num_layers: int) -> float:
    for entry in _json_table("flops.json"):
        config = entry["config"]
        if (
            config["embedding_size"] == embedding_size
            and config["num_layers"] == num_layers
            and config["num_datapoints_max"] == NUM_DATAPOINTS_MAX
            and config["max_features"] == MAX_FEATURES
        ):
            return entry["flops_per_cell"]
    raise ValueError(f"No FLOPs per cell for {embedding_size=}, {num_layers=}")


def parameters_estimator(embedding_size: float, num_layers: float, **_: float) -> int:
    """The parameter count of a config, without querying the benchmark."""
    return _n_params(int(embedding_size), int(num_layers))


def flops_estimator(
    embedding_size: float, num_layers: float, total_cells: float, **_: float
) -> float:
    """The training FLOPs of a config, without querying the benchmark."""
    return _flops_per_cell(int(embedding_size), int(num_layers)) * int(total_cells)


# The search space of the tutorial, which the reference optimum is searched over
CELLS_RANGE = (MIN_CELLS, 2**35)
LR_RANGE = (1e-4, 5e-2)
BATCH_SIZE_RANGE = (16, 256)

# The search for the reference optimum, a compact port of the one of
# `scan.prediction.surrogate_pareto`
REFERENCE_SCREEN = 64
"""Sobol samples of the learning rate and batch size every model size is screened on."""
REFERENCE_REFINED = 8
"""Model sizes whose learning rate and batch size are then refined."""
REFERENCE_POPULATION = 16
REFERENCE_GENERATIONS = 12


@dataclass(frozen=True)
class Reference:
    """The surrogate's best config at a FLOP budget."""

    target_flops: float
    loss: float
    real_nll: float
    n_params: float
    n_data: float
    embedding_size: int
    num_layers: int
    lr: float
    effective_batch_size: int


@cache
def _model_sizes() -> list[tuple[int, int]]:
    """The `(embedding_size, num_layers)` the benchmark counts the FLOPs of."""
    return sorted(
        {
            (entry["config"]["embedding_size"], entry["config"]["num_layers"])
            for entry in _json_table("flops.json")
            if entry["config"]["num_datapoints_max"] == NUM_DATAPOINTS_MAX
            and entry["config"]["max_features"] == MAX_FEATURES
        }
    )


def _hyperparameters(u: np.ndarray) -> dict[str, float]:
    """The learning rate and batch size at `u` of the unit square, both log-scaled."""
    (lr_low, lr_high), (bs_low, bs_high) = LR_RANGE, BATCH_SIZE_RANGE
    lr = np.exp(np.log(lr_low) + u[0] * np.log(lr_high / lr_low))
    return {
        # Clipped, as rounding at the ends would put it just out of the space
        "lr": float(np.clip(lr, lr_low, lr_high)),
        "effective_batch_size": int(
            round(np.exp(np.log(bs_low) + u[1] * np.log(bs_high / bs_low)))
        ),
    }


def _surrogate_losses(sizes: list[dict], us: np.ndarray) -> np.ndarray:
    """The surrogate's loss of each model size of `sizes` with the learning rate and
    batch size at each of `us` (sizes x samples x 2), in one batch.

    Unrounded, as `scan.prediction.surrogate_pareto` takes it: `query_many` rounds to
    3 decimals, which ties the configs the search has to tell apart.
    """
    configs = [
        build_config(**size, **_hyperparameters(u))
        for size, samples in zip(sizes, us, strict=True)
        for u in samples
    ]
    bench = get_benchmark()
    features = bench._configs_to_surrogate_matrix(configs)  # noqa: SLF001
    losses = bench.performance_surrogate.predict_with_uncertainty(features)["mean"]
    return np.asarray(losses, dtype=float).reshape(us.shape[:2])


def _refine(
    sizes: list[dict], best_u: np.ndarray, best_loss: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A (1+lambda) evolution strategy on the learning rate and batch size of every
    model size at once. The step grows after a generation that improves a size and
    shrinks after one that does not, so it settles into narrow optima.
    """
    best_u, best_loss = best_u.copy(), best_loss.copy()
    step = np.full(len(sizes), 0.15)
    for _ in range(REFERENCE_GENERATIONS):
        noise = rng.standard_normal((len(sizes), REFERENCE_POPULATION, 2))
        us = np.clip(best_u[:, None] + step[:, None, None] * noise, 0.0, 1.0)
        losses = _surrogate_losses(sizes, us)
        winner = losses.argmin(axis=1)
        generation_best = losses[np.arange(len(sizes)), winner]
        improved = generation_best < best_loss
        best_u[improved] = us[improved, winner[improved]]
        best_loss[improved] = generation_best[improved]
        step = np.where(improved, step * 1.5, step * 0.5)
    return best_u, best_loss


@cache
def reference_optimum(target_flops: float, seed: int = 0) -> Reference:
    """The surrogate's best config at `target_flops`, over the search space of the
    tutorial: its validation loss, its NLL on real datasets, and its config.

    The optimizers are scored on the surrogate, so its own optimum is the truth they
    are measured against, as `scan.prediction.surrogate_pareto` computes it. Every
    model size is trained on the cells that spend the budget, and its learning rate
    and batch size are screened on Sobol samples; those of the best sizes are then
    refined by an evolution strategy.
    """
    from scipy.stats import qmc

    sizes = []
    for embedding_size, num_layers in _model_sizes():
        cells = int(target_flops / _flops_per_cell(embedding_size, num_layers))
        if CELLS_RANGE[0] <= cells <= CELLS_RANGE[1]:
            sizes.append(
                {
                    "embedding_size": embedding_size,
                    "num_layers": num_layers,
                    "total_cells": cells,
                }
            )
    if not sizes:
        raise ValueError(f"No model size of the space can spend {target_flops:.2e} FLOPs")

    rng = np.random.default_rng(seed)
    screen = qmc.Sobol(d=2, scramble=True, seed=rng).random(REFERENCE_SCREEN)
    losses = _surrogate_losses(sizes, np.broadcast_to(screen, (len(sizes), *screen.shape)))
    best_u, best_loss = screen[losses.argmin(axis=1)], losses.min(axis=1)

    refined = np.argsort(best_loss)[:REFERENCE_REFINED]
    best_u[refined], best_loss[refined] = _refine(
        [sizes[i] for i in refined], best_u[refined], best_loss[refined], rng
    )
    winner = int(best_loss.argmin())
    config = {**sizes[winner], **_hyperparameters(best_u[winner])}

    downstream = get_benchmark(DOWNSTREAM["real_nll"]).query(build_config(**config))
    return Reference(
        target_flops=target_flops,
        loss=float(best_loss[winner]),
        real_nll=float(downstream["predictions"]["mean"]),
        n_params=float(parameters_estimator(**config)),
        n_data=float(config["total_cells"]),
        embedding_size=config["embedding_size"],
        num_layers=config["num_layers"],
        lr=config["lr"],
        effective_batch_size=config["effective_batch_size"],
    )


def set_seed(seed: int = 0) -> None:
    """Seed the generators the optimizers draw from."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
