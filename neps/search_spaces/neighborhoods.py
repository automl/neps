from __future__ import annotations

from typing import TypeVar

import numpy as np

from neps.search_spaces.domain import Domain
from neps.utils.types import Arr, f64, i64

V = TypeVar("V", f64, i64)

UNIQUE_NEIGHBOR_GENERATOR_N_RETRIES = 8
UNIQUE_NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER = 4

NON_UNIQUE_NEIGHBORS_N_RETRIES = 8
NON_UNIQUE_NEIGHBORS_SAMPLE_MULTIPLIER = 4

# Small enough but prevents needing to keep re-allocating temporary memory
# 50 * 8 = 400 bytes
_SMALL = 50
_SMALL_CACHED_ARANGE = np.arange(_SMALL, dtype=i64)


def unorded_finite_neighbors(
    pivot: V,
    domain: Domain[V],
    *,
    n: int,
    seed: np.random.Generator,
) -> Arr[V]:
    N = domain.cardinality
    assert N is not None, "Domain must be finite."
    if N <= _SMALL:
        full_range = _SMALL_CACHED_ARANGE[: domain.cardinality]
    else:
        full_range = np.arange(N, dtype=i64)

    range_domain = Domain.indices(N)
    _pivot = range_domain.cast(pivot, frm=domain)

    left = full_range[:_pivot]
    right = full_range[_pivot + 1 :]
    _range = np.concatenate((left, right))

    seed.shuffle(_range)

    return domain.cast(_range[:n], frm=range_domain)


def neighbors(
    pivot: V,
    domain: Domain[V],
    *,
    n: int,
    std: float,
    seed: np.random.Generator,
    n_retries: int = NON_UNIQUE_NEIGHBORS_N_RETRIES,
    sample_multiplier: int = NON_UNIQUE_NEIGHBORS_SAMPLE_MULTIPLIER,
) -> Arr[V]:
    """Create a neighborhood of `n` neighbors around `pivot` with a normal distribution.

    If you need unique neighbors, you should use
    [`unique_neighborhood`][neps.search_spaces.neighborhoods.unique_neighborhood].

    !!! tip

        [`unique_neighborhood`][neps.search_spaces.neighborhoods.unique_neighborhood]
        is quite expensive in certain situations as it has to repeatedly sample and check
        for uniqueness. If you can afford duplicates, use this function instead.

        If [`domain.cardinality == None`][neps.search_spaces.domain.Domain.cardinality],
        and you can afford an infentesimally small percentage change of duplicates,
        you should use this function instead.

    !!! warning

        It is up to the caller to ensure that the pivot lies within the domain,
        including at one of the bins if the domain is quantized.

    Args:
        pivot: The center of the neighborhood.
        domain: The domain to get neighbors from.
        n: The number of neighbors to generate.
        std: The standard deviation of the normal distribution.
        seed: The random seed to use.
        n_retries:
            The number of retries to attempt to generate unique neighbors.
            Each retry increases the standard deviation of the normal distribution to
            prevent rejection sampling from failing.
        sample_multiplier:
            A multiplier which multiplies by `n` to determine the number of samples to
            generate for try. By oversampling, we prevent having to repeated calls to
            sampling. This prevents having to do more rounds of sampling when too many
            samples are out of bounds, useful for when the `pivot` is near the bounds.

            Tuning this may be beneficial in unique circumstances, however we advise
            leaving this as a default.

    Returns:
        An array of `n` neighbors around `pivot`.
    """
    # Generate batches of n * BUFFER_MULTIPLIER candidates, filling the above
    # buffer until we have enough valid candidates.
    # We should not overflow as the buffer
    offset = 0
    SAMPLE_SIZE = n * sample_multiplier
    BUFFER_SIZE = (n + 1) * sample_multiplier

    # We extend the range of stds to try to find neighbors
    neighbors: Arr[V] = np.empty(BUFFER_SIZE, dtype=domain.dtype)
    stds = np.linspace(std, 1.0, n_retries + 1, endpoint=True)

    lower = domain.lower
    upper = domain.upper
    range_size = upper - lower
    sample_domain = Domain.float(lower, upper)

    for _std in stds:
        candidates = seed.normal(pivot, _std * range_size, size=(SAMPLE_SIZE,))

        bounded_candidates = candidates[(candidates >= lower) & (candidates <= upper)]
        maybe_valid = domain.cast(bounded_candidates, frm=sample_domain)

        # High chance of overlap with original point if there's a finite amount of
        # possible elements
        if domain.cardinality is not None:
            valid = maybe_valid[maybe_valid != pivot]
        else:
            valid = maybe_valid

        n_candidates = len(valid)
        neighbors[offset : offset + n_candidates] = valid
        offset += n_candidates

        if offset >= n:
            return neighbors[:n]

    raise ValueError(
        f"Failed to find enough neighbors with {n_retries} retries."
        f" Given {n} neighbors, we only found {offset}."
        f" The `Normals` for sampling neighbors were"
        f" Normal(mu={pivot}, sigma={list(stds)})"
        f" which were meant to find vectorized neighbors of the vector {pivot},"
        " which was expected to be in the range"
        f" ({lower}, {lower}).",
    )


def unique_neighborhood(
    pivot: V,
    domain: Domain[V],
    *,
    n: int,
    seed: np.random.Generator,
    std: float,
    n_retries: int = UNIQUE_NEIGHBOR_GENERATOR_N_RETRIES,
    sample_multiplier: int = UNIQUE_NEIGHBOR_GENERATOR_SAMPLE_MULTIPLIER,
) -> Arr[V]:
    """Create a neighborhood of `n` neighbors around `pivot` with a normal distribution.

    The neighborhood is created by sampling from a normal distribution centered around
    `pivot` with a standard deviation of `std`. The samples are then quantized to the
    range `[lower, upper]` with `bins` bins. The number of samples is `n`.

    !!! tip

        [`unique_neighborhood`][neps.search_spaces.neighborhoods.unique_neighborhood]
        is quite expensive in certain situations as it has to repeatedly sample and check
        for uniqueness. If you can afford duplicates, use this function instead.

        If [`domain.cardinality == None`][neps.search_spaces.domain.Domain.cardinality],
        and you can afford an infentesimally small percentage change of duplicates,
        you should use [`neighbors`][neps.search_spaces.neighborhoods.neighbors] instead.

    !!! warning

        If there are not enough unique neighbors to sample from, the function will
        return less than `n` neighbors.

    !!! warning

        It is up to the caller to ensure that the pivot lies within the domain,
        including at one of the bins if the domain is quantized.


    Args:
        pivot: The center of the neighborhood.
        domain: The domain to get neighbors from.
        n: The number of neighbors to generate.
        std: The standard deviation of the normal distribution.
        seed: The random seed to use.
        n_retries:
            The number of retries to attempt to generate unique neighbors.
            Each retry increases the standard deviation of the normal distribution to prevent
            rejection sampling from failing.
        sample_multiplier:
            A multiplier which multiplies by `n` to determine the number of samples to
            generate for try. By oversampling, we prevent having to repeated calls to
            both sampling and unique checking.

            However, oversampling makes a tradeoff when the `std` is not high enough to
            generate `n` unique neighbors, effectively sampling more of the same duplicates.

            Tuning this may be beneficial in unique circumstances, however we advise leaving
            this as a default.

    Returns:
        An array of `n` neighbors around `pivot`, or less than `n` if not enough unique
        neighbors could be generated.
    """  # noqa: E501
    # Different than other neighborhoods as it's unnormalized and
    # the quantization is directly integers.
    assert n < 1000000, "Can only generate less than 1 million neighbors."
    assert 0 < std < 1.0, "Standard deviation must be in the range (0, 1)."
    lower = domain.lower
    upper = domain.upper

    # In the easiest case, we have a domain with finite elements and we need
    # more neighbors than are possible. We then generate all of them.
    # We can do this simply with a range and removing the pivot.
    if domain.cardinality is not None and n >= domain.cardinality - 1:
        range_domain = Domain.indices(domain.cardinality)
        int_pivot = range_domain.cast(pivot, frm=domain)

        if int_pivot == 0:
            _range = np.arange(1, domain.cardinality, dtype=i64)
            return domain.cast(_range, frm=range_domain)

        if int_pivot == domain.cardinality - 1:
            _range = np.arange(0, domain.cardinality - 1, dtype=i64)
            return domain.cast(_range, frm=range_domain)

        left = np.arange(0, int_pivot, dtype=i64)
        right = np.arange(int_pivot + 1, domain.cardinality, dtype=i64)
        _range = np.concatenate((left, right))

        return domain.cast(_range, frm=range_domain)

    # Otherwise, we use a repeated sampling strategy where we slowly increase the
    # std of a normal, centered on `center`, slowly expanding `std` such that
    # rejection won't fail.

    # We set up a buffer that can hold the number of neighbors we need, plus some
    # extra excess from sampling, preventing us from having to reallocate memory.
    # We also include the initial value in the buffer, as we will remove it later.
    SAMPLE_SIZE = n * sample_multiplier
    BUFFER_SIZE = n * (sample_multiplier + 1)
    neighbors = np.empty(BUFFER_SIZE + 1, dtype=domain.dtype)
    neighbors[0] = pivot
    offset = 1  # Indexes into current progress of filling buffer
    stds = np.linspace(std, 1.0, n_retries + 1, endpoint=True)
    sample_domain = Domain.float(lower, upper)

    range_size = upper - lower
    for _std in stds:
        # Generate candidates in vectorized space
        candidates = seed.normal(pivot, _std * range_size, size=SAMPLE_SIZE)
        valid = (candidates >= lower) & (candidates <= upper)

        candidates = domain.cast(x=candidates[valid], frm=sample_domain)

        # Find new unique neighbors
        uniq = np.unique(candidates)
        new_uniq = np.setdiff1d(uniq, neighbors[:offset], assume_unique=True)

        n_new_unique = len(new_uniq)
        neighbors[offset : offset + n_new_unique] = new_uniq
        offset += n_new_unique

        # We have enough neighbors, we can stop
        if offset - 1 >= n:
            # Ensure we don't include the initial value point
            return neighbors[1 : n + 1]

    raise ValueError(
        f"Failed to find enough neighbors with {n_retries} retries."
        f" Given {n=} neighbors to generate, we only found {offset - 1}."
        f" The normal's for sampling neighbors were Normal({pivot}, {list(stds)})"
        f" which were meant to find neighbors of {pivot}. in the range"
        f" ({lower}, {upper}).",
    )
