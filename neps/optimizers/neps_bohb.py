"""BOHB for NePS spaces: Hyperband brackets + the new BO's kernel surrogate.

This implements the "new BOHB" arm of the 2x2 comparison:

* classic BO (`bayesian_optimization`): GP on all trials, no brackets
* classic BOHB (`bohb`): KDE per budget in Hyperband brackets
* new BO (`neps_bayesian_optimization`): KRR + pluggable kernel on all
  trials, no brackets
* new BOHB (`neps_bohb`): **KRR + pluggable kernel on all trials, in
  Hyperband brackets**

The surrogate machinery is identical to `NepsBayesianOptimization`
(`KernelSurrogateModel` with a pluggable `kernel_fn` over neps config dicts,
EI/LogEI acquisition over freshly resolved candidates), so the only
difference between new BO and new BOHB is the multi-fidelity bracket
structure. The only difference between classic BOHB and new BOHB is the
surrogate (KDE vs kernel KRR).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Literal

import numpy as np
import pandas as pd

from neps.optimizers.models.neps_gp import (
    KernelSurrogateModel,
    _expected_improvement,
    _greedy_batch_acquisition,
    _hamming_kernel,
    _log_expected_improvement,
)
from neps.space.neps_spaces import neps_space
from neps.space.neps_spaces.neps_space import NepsCompatConverter, resolve
from neps.space.neps_spaces.sampling import (
    OnlyPredefinedValuesSampler,
    RandomSampler,
)

if TYPE_CHECKING:
    from neps.space.neps_spaces.parameters import PipelineSpace

logger = logging.getLogger(__name__)


@dataclass
class NePSBOHBSampler:
    """A kernel-surrogate sampler for NePS-space bracket optimizers.

    Implements the per-rung sampling of BOHB (Falkner et al., ICML 2018)
    but with the new BO's surrogate instead of the TPE-style KDE:

    1. With probability `random_fraction` (rho in the paper), sample a
       random configuration at the rung's fidelity.
    2. Otherwise, fit `KernelSurrogateModel` on **all** completed trials
       (same as the new BO — fidelity is part of the config dict and thus
       part of the kernel input), sample `acqu_sampling_density` candidates
       at the rung's fidelity, and return the candidate with the best
       acquisition value.
    """

    space: PipelineSpace
    """The pipeline space to sample from."""

    rung_to_fid: Mapping[int, int | float]
    """The mapping from rung to fidelity value."""

    fid_name: str
    """The raw fidelity attribute name (without the `ENVIRONMENT__` prefix)."""

    kernel_function: Callable = _hamming_kernel
    """The kernel function `kernel_fn(config_i: dict, config_j: dict) -> float`.
    Same contract as `neps_bayesian_optimization`'s `kernel_function`."""

    acqu_function: Callable = _expected_improvement
    """The acquisition function `(pred_mean, pred_var, best_y) -> float`."""

    acqu_sampling_density: int = 1000
    """The number of candidates to sample and score per acquisition."""

    use_batch_acquisition: bool = False
    """Whether to use greedy batch-aware acquisition (as in the new BO)."""

    krr_alpha: float = 1e-6
    """The KRR regularization parameter."""

    random_fraction: float = 1 / 3
    """The fraction of configurations sampled uniformly at random (rho in
    the BOHB paper)."""

    min_points_in_model: int = 4
    """Fit the surrogate only once more than this many trials are completed.
    Default 4 matches the new BO's `initial_design_size=5`."""

    surrogate: Literal["krr", "gp"] = "krr"
    """The surrogate model to use. One of:
    - "krr": KernelSurrogateModel with the given kernel_function (the
      default, unchanged behavior).
    - "gp": GPSurrogateModel — exact GP over the given kernel_function."""

    hp_tuning: Literal["none", "loo_cv", "mll", "mll_noise_only"] = "none"
    """How the surrogate's hyperparameters are set each fit. Same contract
    as `neps_bayesian_optimization`'s `hp_tuning` ("none", "loo_cv",
    "mll", "mll_noise_only")."""

    sampler_cache_generator: Callable | None = None
    """Optional callable `(n_to_sample=1, repeat=False) -> list[tuple]` that
    generates candidate pipelines (e.g. from a cache pool). Same contract as
    the new BO's `sampler_cache_generator`."""

    _random_sampler: RandomSampler = field(
        default_factory=lambda: RandomSampler(predefined_samplings={}),
        init=False,
    )
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(),
        init=False,
    )

    def __post_init__(self) -> None:
        if isinstance(self.acqu_function, str):
            match self.acqu_function:
                case "EI":
                    self.acqu_function = _expected_improvement
                case "LogEI":
                    self.acqu_function = _log_expected_improvement
                case _:
                    raise ValueError(
                        f"Unsupported acquisition function: {self.acqu_function}"
                    )
        if not 0 < self.random_fraction < 1:
            raise ValueError(
                f"random_fraction should be in (0, 1), got {self.random_fraction}"
            )

    def _resolve_at_rung(self, rung: int) -> dict[str, Any]:
        """Resolve a fresh random candidate with the fidelity pinned to the rung."""
        _resolved, context = resolve(
            pipeline=self.space,
            domain_sampler=self._random_sampler,
            environment_values={self.fid_name: self.rung_to_fid[rung]},
        )
        return dict(NepsCompatConverter.to_neps_config(context))

    def _candidate_from_cache_generator(self, rung: int) -> dict[str, Any]:
        """Generate a candidate via the cache generator and pin its fidelity.

        The cache generator returns pipelines at arbitrary fidelities; we
        re-resolve with `OnlyPredefinedValuesSampler` to pin the fidelity to
        the rung (same mechanism as `_NePSBracketOptimizer._convert_to_another_rung`).
        """
        chosen = self.sampler_cache_generator(n_to_sample=1, repeat=False)[0]
        _resolved_pipeline, context = chosen
        data = NepsCompatConverter.from_neps_config(
            NepsCompatConverter.to_neps_config(context)
        )
        _resolved, recontext = resolve(
            pipeline=self.space,
            domain_sampler=OnlyPredefinedValuesSampler(
                predefined_samplings=data.predefined_samplings,
            ),
            environment_values={
                **data.environment_values,
                self.fid_name: self.rung_to_fid[rung],
            },
        )
        return dict(NepsCompatConverter.to_neps_config(recontext))

    def sample_config(self, table: pd.DataFrame, rung: int) -> dict[str, Any]:  # noqa: C901, PLR0912
        """Sample a configuration for the given rung.

        Args:
            table: The table of all trials (neps configs + perf).
            rung: The rung to sample for; its fidelity is pinned on all
                candidates.

        Returns:
            The sampled neps config dict (fidelity pinned to the rung).
        """
        completed = table[table["perf"].notna()]

        # 1. Random fraction, or not enough data to fit the surrogate
        if (
            len(completed) == 0
            or len(completed) <= self.min_points_in_model
            or self._rng.random() < self.random_fraction
        ):
            return self._resolve_at_rung(rung)

        # 2. Fit the surrogate on ALL completed trials (parity with the new
        # BO). `surrogate` selects the model/kernel, `hp_tuning` how its
        # hyperparameters are set. The fit time is recorded in
        # self.surrogate_fit_times and, if the NEPS_SURROGATE_TIMES_FILE
        # env var is set, appended to that file (one float per line) so the
        # experiment runner can persist it.
        configs = [dict(row["config"]) for _, row in completed.iterrows()]
        y = completed["perf"].to_numpy(dtype=np.float64)
        import os as _os
        import time as _time

        _fit_t0 = _time.perf_counter()
        from neps.optimizers.neps_bayesian_optimization import (
            build_kernel_surrogate,
        )

        surrogate = build_kernel_surrogate(
            training_configs=configs,
            training_y=y,
            kernel_fn=self.kernel_function,
            surrogate=self.surrogate,
            hp_tuning=self.hp_tuning,
            krr_alpha=self.krr_alpha,
        )
        fit_time = _time.perf_counter() - _fit_t0
        if not hasattr(self, "surrogate_fit_times"):
            self.surrogate_fit_times = []
        self.surrogate_fit_times.append(fit_time)
        times_file = _os.environ.get("NEPS_SURROGATE_TIMES_FILE")
        if times_file:
            with open(times_file, "a") as f:
                f.write(f"{fit_time}\n")

        best_y = float(np.min(y))

        # 3. Sample and score candidates at the rung's fidelity
        candidate_configs: list[dict[str, Any]] = []
        candidate_scores: list[tuple[float, float]] = []
        for _ in range(self.acqu_sampling_density):
            if self.sampler_cache_generator is not None:
                config = self._candidate_from_cache_generator(rung)
            else:
                config = self._resolve_at_rung(rung)
            pred_mean, pred_var = surrogate.predict(config)
            candidate_configs.append(config)
            candidate_scores.append((pred_mean, pred_var))

        # 4. Acquisition and selection (same as the new BO)
        if self.use_batch_acquisition:
            pred_means = np.array([s[0] for s in candidate_scores])
            pred_vars = np.array([s[1] for s in candidate_scores])
            acqu_scores = _greedy_batch_acquisition(
                pred_means,
                pred_vars,
                best_y,
                kernel_fn=self.kernel_function,
                candidate_configs=candidate_configs,
                acqu_func=self.acqu_function,
            )
        else:
            acqu_scores = np.array(
                [
                    self.acqu_function(mean, var, best_y)
                    for mean, var in candidate_scores
                ]
            )

        best_idx = int(np.argmax(acqu_scores))
        return candidate_configs[best_idx]
