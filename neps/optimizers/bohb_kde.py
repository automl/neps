"""Kernel density estimators for the BOHB sampler.

This module implements the KDE machinery used by BOHB
(Falkner et al., ICML 2018, https://arxiv.org/abs/1807.01774).

The paper (Appendix D) uses the ``statsmodels`` ``KDEMultivariate`` with
Scott's rule of thumb bandwidth estimation. We re-implement the same
kernels here to avoid a hard dependency on ``statsmodels``:

* Gaussian kernel for continuous (and integer) parameters
* Aitchison-Aitken kernel for categorical parameters

All data is expected to lie in the unit interval ``[0, 1]``, which is
guaranteed by the NePS ``ConfigEncoder`` (numericals are min-max
normalized, categoricals are encoded as binned floats in ``[0, 1]``).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)

# The bandwidth factor used to widen the bandwidth of the good KDE when
# proposing candidates to optimize the acquisition function (Eq. for l'(x)
# in the paper). See `BOHBSampler` for more.
DEFAULT_BANDWIDTH_FACTOR = 3.0
DEFAULT_MIN_BANDWIDTH = 1e-3


def scott_bandwidths(data: np.ndarray) -> np.ndarray:
    """Scott's rule of thumb bandwidth estimation, as used by statsmodels.

    This is the same estimation procedure used by the statsmodels
    ``KDEMultivariate`` with ``bw='normal_reference'``, which is what the
    BOHB paper (Appendix D) and the reference implementation use.

    For each dimension, the bandwidth is:

        bw = 1.059 * min(std, IQR / 1.34) * n^(-1/5)

    where `IQR` is the interquartile range. If the IQR is zero (e.g. all
    values identical), the standard deviation is used instead.

    Args:
        data: A 2d array of shape `(n, d)` with values in [0, 1].

    Returns:
        A 1d array of shape `(d,)` with the bandwidths.
    """
    n = data.shape[0]
    sigmas = np.std(data, ddof=1, axis=0)
    q75, q25 = np.percentile(data, [75, 25], axis=0)
    iqrs = q75 - q25
    # statsmodels uses min(std, IQR / 1.349), falling back to std when the
    # IQR is zero (which happens when > 75% of the data is identical).
    iqr_based = iqrs / 1.349
    fallback = np.where(iqr_based > 0, iqr_based, sigmas)
    return 1.059 * np.minimum(sigmas, fallback) * n ** (-1 / 5)


class _KDE:
    """A factorized multivariate KDE over data in the unit interval.

    The kernel is a product of per-dimension kernels:

    * Gaussian for continuous dimensions
    * Aitchison-Aitken for categorical dimensions

    This mirrors the ``statsmodels`` ``KDEMultivariate`` used in the paper,
    with the addition of boundary correction for the Gaussian kernel
    (re-weighting points close to the boundary, as done in the HpBandSter
    reference implementation).
    """

    def __init__(
        self,
        data: np.ndarray,
        bandwidths: np.ndarray,
        is_categorical: np.ndarray,
        *,
        min_bandwidth: float = DEFAULT_MIN_BANDWIDTH,
        fix_boundary: bool = True,
    ) -> None:
        """Initialize the KDE.

        Args:
            data: A 2d array of shape `(n, d)` with values in [0, 1].
            bandwidths: A 1d array of shape `(d,)` with the bandwidths.
            is_categorical: A boolean array of shape `(d,)` indicating which
                dimensions are categorical.
            min_bandwidth: The minimum bandwidth to use for any dimension.
            fix_boundary: Whether to re-weight points close to the boundary
                of the unit interval for the Gaussian kernel.
        """
        self.data = data
        self.is_categorical = is_categorical
        self.min_bandwidth = min_bandwidth
        self.fix_boundary = fix_boundary
        self.bandwidths = np.clip(
            bandwidths,
            min_bandwidth,
            # Categorical bandwidths must be < 1 for the Aitchison-Aitken
            # kernel to be a valid distribution, and we keep continuous
            # bandwidths below 0.5 to avoid degenerate densities.
            np.where(is_categorical, 0.999, 0.5),
        )
        self._weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        if not self.fix_boundary or not np.any(~self.is_categorical):
            return np.ones(len(self.data))

        # Boundary correction for the Gaussian kernel: re-weight each data
        # point by the inverse of the mass of its truncated Gaussian kernel
        # that falls inside the unit interval, per continuous dimension.
        cont_dims = np.where(~self.is_categorical)[0]
        weights = np.ones(len(self.data))
        for i in cont_dims:
            bw = self.bandwidths[i]
            d = self.data[:, i]
            weights *= 2.0 / (_norm_cdf((1 - d) / bw) + _norm_cdf(d / bw))
        return weights

    def pdf(self, x_test: np.ndarray) -> np.ndarray:
        """Evaluate the probability density at the given points.

        Args:
            x_test: A 2d array of shape `(n, d)` with values in [0, 1].

        Returns:
            A 1d array of shape `(n,)` with the density values.
        """
        x_test = np.atleast_2d(x_test)
        n_data = len(self.data)
        n_test = len(x_test)

        pdfs = np.ones((n_test, n_data))
        for i in range(x_test.shape[1]):
            bw = self.bandwidths[i]
            if self.is_categorical[i]:
                # Aitchison-Aitken kernel
                # p(x | data) = (1 - bw) if x == data_point else bw / (num_values - 1)
                # We don't know `num_values` here, so we use the number of
                # unique values in the data, with a fallback of 2.
                num_values = max(len(np.unique(self.data[:, i])), 2)
                same = np.isclose(
                    x_test[:, i, None], self.data[None, :, i], atol=1e-8
                )
                kernel = np.where(same, 1 - bw, bw / (num_values - 1))
            else:
                # Gaussian kernel
                distances = x_test[:, i, None] - self.data[None, :, i]
                kernel = np.exp(-0.5 * (distances / bw) ** 2) / (
                    np.sqrt(2 * np.pi) * bw
                )
            pdfs *= kernel

        return (pdfs * self._weights[None, :]).sum(axis=1)

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample points from the KDE.

        For each sample, a data point is chosen uniformly at random as the
        kernel center, then each dimension is perturbed according to its
        kernel:

        * Gaussian: truncated normal around the center
        * Aitchison-Aitken: categorical resampling

        Args:
            num_samples: The number of samples to draw.
            rng: The random number generator to use.

        Returns:
            A 2d array of shape `(num_samples, d)` with values in [0, 1].
        """
        n_data, d = self.data.shape
        centers = self.data[rng.integers(0, n_data, size=num_samples)]
        samples = centers.copy()

        for i in range(d):
            bw = self.bandwidths[i]
            if self.is_categorical[i]:
                # Aitchison-Aitken sampling: with probability (1 - bw) keep
                # the center value, otherwise sample uniformly from the
                # other values.
                num_values = max(len(np.unique(self.data[:, i])), 2)
                keep = rng.random(num_samples) < (1 - bw)
                other = rng.integers(0, num_values, size=num_samples)
                # Map the random integer into the domain, skipping the
                # center value (approximately, via modular arithmetic).
                shifted = (centers[:, i] * num_values + other).astype(int) % num_values
                new_vals = shifted / max(num_values - 1, 1)
                samples[:, i] = np.where(keep, centers[:, i], new_vals)
            else:
                # Truncated normal around the center
                bw_eff = bw
                a = (0 - centers[:, i]) / bw_eff
                b = (1 - centers[:, i]) / bw_eff
                samples[:, i] = _truncnorm_sample(a, b, centers[:, i], bw_eff, rng)

        return np.clip(samples, 0.0, 1.0)


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    from scipy.stats import norm

    return norm.cdf(x)


def _truncnorm_sample(
    a: np.ndarray,
    b: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a truncated normal distribution, vectorized.

    Uses inverse CDF sampling, with a fallback to clipping for extreme
    tails where the inverse CDF is numerically unstable.
    """
    from scipy.stats import norm, truncnorm

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    loc = np.asarray(loc, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)

    # Inverse CDF sampling
    u = rng.random(len(a))
    alpha = norm.cdf(a)
    beta = norm.cdf(b)
    # Guard against degenerate intervals
    beta = np.maximum(beta, alpha + 1e-12)
    z = norm.ppf(alpha + u * (beta - alpha))
    samples = loc + z * scale

    # Fallback for numerical instability in the tails
    bad = ~np.isfinite(samples)
    if np.any(bad):
        samples[bad] = np.clip(loc[bad], 0.0, 1.0)

    return np.clip(samples, 0.0, 1.0)


def fit_kde(
    data: np.ndarray,
    is_categorical: np.ndarray,
    *,
    min_bandwidth: float = DEFAULT_MIN_BANDWIDTH,
    fix_boundary: bool = True,
) -> _KDE:
    """Fit a KDE to the given data using Scott's rule of thumb.

    Args:
        data: A 2d array of shape `(n, d)` with values in [0, 1].
        is_categorical: A boolean array of shape `(d,)`.
        min_bandwidth: The minimum bandwidth to use.
        fix_boundary: Whether to apply boundary correction.

    Returns:
        The fitted `_KDE`.
    """
    bandwidths = scott_bandwidths(data)
    return _KDE(
        data=data,
        bandwidths=bandwidths,
        is_categorical=is_categorical,
        min_bandwidth=min_bandwidth,
        fix_boundary=fix_boundary,
    )
