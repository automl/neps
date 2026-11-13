"""The BOHB sampler: a TPE-style KDE sampler for bracket optimizers.

Implements the BO component of BOHB (Falkner et al., ICML 2018,
https://arxiv.org/abs/1807.01774), Algorithm 2:

1. With probability `random_fraction` (rho in the paper), sample uniformly.
2. Otherwise, use the model for the largest budget (rung) with at least
   `min_points_in_model + 2` observations.
3. Split the observations at that budget into good (`l(x)`) and bad (`g(x)`)
   sets using the quantile `q` (`top_n_percent`).
4. Sample `num_samples` candidates from `l'(x)` (the good KDE with widened
   bandwidths) and pick the one maximizing `l'(x) / g(x)`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from neps.optimizers.bohb_kde import (
    DEFAULT_BANDWIDTH_FACTOR,
    DEFAULT_MIN_BANDWIDTH,
    _KDE,
    fit_kde,
)
from neps.space import ConfigEncoder

if TYPE_CHECKING:
    from neps.space.parameters import Parameter

logger = logging.getLogger(__name__)


@dataclass
class BOHBSampler:
    """A TPE-style sampler using kernel density estimators, as in BOHB.

    For each rung (budget), a pair of KDEs is fit on the best
    `top_n_percent` (good, `l(x)`) and worst (bad, `g(x)`) configurations
    evaluated at that budget. New configurations are proposed by sampling
    from the good KDE with widened bandwidths and maximizing the ratio
    `l'(x) / g(x)`, which is equivalent to maximizing EI under the TPE
    formulation (Bergstra et al., 2011).
    """

    parameters: Mapping[str, Parameter]
    """The parameters to consider."""

    encoder: ConfigEncoder
    """The encoder used to encode/decode configurations into tensors."""

    min_points_in_model: int | None = None
    """Minimum number of observations needed to fit a KDE.

    If `None`, defaults to `d + 1` where `d` is the number of
    hyperparameters, as in the paper.
    """

    top_n_percent: int = 15
    """The percentage of best configurations used to fit the good KDE (`q`)."""

    num_samples: int = 64
    """The number of candidates sampled from `l'(x)` to optimize the
    acquisition function (`N_s`)."""

    random_fraction: float = 1 / 3
    """The fraction of configurations sampled uniformly at random (`rho`)."""

    bandwidth_factor: float = DEFAULT_BANDWIDTH_FACTOR
    """The factor by which to widen the bandwidths of the good KDE when
    proposing candidates (`bw`)."""

    min_bandwidth: float = DEFAULT_MIN_BANDWIDTH
    """The minimum bandwidth used for any dimension."""

    # Internal state: per-rung KDE models
    _kde_models: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(), init=False
    )

    def __post_init__(self) -> None:
        if self.min_points_in_model is None:
            self.min_points_in_model = len(self.parameters) + 1
        if self.min_points_in_model < len(self.parameters) + 1:
            logger.warning(
                "min_points_in_model (%s) is smaller than the number of"
                " hyperparameters + 1 (%s). Using the latter.",
                self.min_points_in_model,
                len(self.parameters) + 1,
            )
            self.min_points_in_model = len(self.parameters) + 1
        if not 0 < self.random_fraction < 1:
            raise ValueError(
                f"random_fraction should be in (0, 1), got {self.random_fraction}"
            )
        if not 0 < self.top_n_percent < 100:
            raise ValueError(
                f"top_n_percent should be in (0, 100), got {self.top_n_percent}"
            )

    @property
    def _is_categorical(self) -> np.ndarray:
        return np.array(
            [
                self.encoder.domain_of[name].is_categorical
                for name in self.encoder.transformers
            ],
            dtype=bool,
        )

    def _encode(self, configs: list[dict[str, Any]]) -> np.ndarray:
        tensor = self.encoder.encode(configs, dtype=torch.float64)
        return tensor.numpy()

    def _decode(self, vector: np.ndarray) -> dict[str, Any]:
        tensor = torch.from_numpy(vector).unsqueeze(0)
        return self.encoder.decode_one(tensor)

    def _fit_kdes_for_rung(self, rung: int, table: pd.DataFrame) -> bool:
        """Fit the good/bad KDE pair for a rung, if enough data is available.

        Args:
            rung: The rung to fit the KDEs for.
            table: The table of trials, with `config` and `perf` columns.

        Returns:
            Whether the KDEs were (re-)fit.
        """
        rung_table = table[table.index.get_level_values("rung") == rung]
        completed = rung_table[rung_table["perf"].notna()]
        if len(completed) <= self.min_points_in_model:  # type: ignore[operator]
            return False

        configs = completed["config"].tolist()
        perfs = completed["perf"].to_numpy(dtype=np.float64)
        order = np.argsort(perfs)
        configs = [configs[i] for i in order]
        perfs = perfs[order]

        n_good = max(self.min_points_in_model, (self.top_n_percent * len(configs)) // 100)  # type: ignore[operator]
        n_bad = max(self.min_points_in_model, ((100 - self.top_n_percent) * len(configs)) // 100)  # type: ignore[operator]

        # Guard against degenerate splits (as in the reference implementation)
        if n_good >= len(configs) or n_bad >= len(configs):
            return False

        good_data = self._encode(configs[:n_good])
        bad_data = self._encode(configs[n_good : n_good + n_bad])

        # The reference implementation skips model building if either split
        # has fewer points than dimensions (KDE would be degenerate).
        if good_data.shape[0] <= good_data.shape[1]:
            return False
        if bad_data.shape[0] <= bad_data.shape[1]:
            return False

        is_cat = self._is_categorical
        self._kde_models[rung] = {
            "good": fit_kde(good_data, is_cat, min_bandwidth=self.min_bandwidth),
            "bad": fit_kde(bad_data, is_cat, min_bandwidth=self.min_bandwidth),
        }
        return True

    def _largest_budget_with_model(self, table: pd.DataFrame) -> int | None:
        """The largest rung with at least `min_points_in_model + 1` completed
        observations, as per line 2 of Algorithm 2 in the paper."""
        completed = table[table["perf"].notna()]
        if len(completed) == 0:
            return None
        rung_counts = completed.groupby(level="rung").size()
        eligible = rung_counts[rung_counts > self.min_points_in_model]  # type: ignore[operator]
        if eligible.empty:
            return None
        return int(eligible.index.max())

    def sample_config(self, table: pd.DataFrame, rung: int) -> dict[str, Any]:
        """Sample a configuration for the given rung, following Algorithm 2.

        Args:
            table: The table of all trials that have been run.
            rung: The rung to sample for. Used as the fallback budget when
                no model exists yet; the model for the largest budget with
                enough observations is always preferred, as in the paper.

        Returns:
            The sampled configuration.
        """
        # 1. Random fraction, or no model available yet
        budget = self._largest_budget_with_model(table)
        if budget is None or self._rng.random() < self.random_fraction:
            sampler = torch.distributions.Uniform(0, 1)
            sample = sampler.sample(torch.Size([self.encoder.ndim]))
            return self.encoder.decode_one(sample)

        # 2. Refit the KDEs for the budget being used (cheap, Scott's rule),
        # so the model always reflects the latest observations, as in the
        # reference implementation where models are refit on every new result.
        self._fit_kdes_for_rung(budget, table)
        if budget not in self._kde_models:
            # Refitting failed (e.g. degenerate split), fall back to random
            sampler = torch.distributions.Uniform(0, 1)
            sample = sampler.sample(torch.Size([self.encoder.ndim]))
            return self.encoder.decode_one(sample)

        # 2. Sample candidates from the widened good KDE and maximize l'/g
        kde_good = self._kde_models[budget]["good"]
        kde_bad = self._kde_models[budget]["bad"]

        # l'(x): same KDE as l(x) but with bandwidths multiplied by bw_factor
        widened = _KDE(
            data=kde_good.data,
            bandwidths=kde_good.bandwidths * self.bandwidth_factor,
            is_categorical=kde_good.is_categorical,
            min_bandwidth=self.min_bandwidth,
            fix_boundary=kde_good.fix_boundary,
        )

        candidates = widened.sample(self.num_samples, self._rng)
        l_vals = widened.pdf(candidates)
        g_vals = kde_bad.pdf(candidates)

        # minimize l/g == maximize l/g ratio; guard against zero densities
        ratio = l_vals / np.maximum(g_vals, 1e-32)
        best_idx = int(np.argmax(ratio))
        best_vector = candidates[best_idx]

        # Round categorical columns back to their bin centers before decoding
        is_cat = self._is_categorical
        for i in range(len(is_cat)):
            if is_cat[i]:
                domain = self.encoder.domain_of[
                    list(self.encoder.transformers)[i]
                ]
                n_bins = domain.cardinality
                best_vector[i] = np.round(best_vector[i] * (n_bins - 1)) / (n_bins - 1)

        return self._decode(best_vector)
