"""Functions for working with search spaces."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from neps.search_spaces.domain import UNIT_FLOAT_DOMAIN, Domain
from neps.search_spaces.parameter import Parameter, ParameterWithPrior
from neps.search_spaces.search_space import SearchSpace

if TYPE_CHECKING:
    from neps.search_spaces.encoding import ConfigEncoder

logger = logging.getLogger(__name__)


def pairwise_dist(
    x: torch.Tensor,
    encoder: ConfigEncoder,
    *,
    numerical_ord: int = 2,
    categorical_ord: int = 0,
    dtype: torch.dtype = torch.float64,
    square_form: bool = False,
) -> torch.Tensor:
    """Compute the pairwise distance between rows of a tensor.

    Will sum the results of the numerical and categorical distances.
    The encoding will be normalized such that all numericals lie within the unit
    cube, and categoricals will by default, have a `p=0` norm, which is equivalent
    to the Hamming distance.

    Args:
        x: A tensor of shape `(N, ncols)`.
        encoder: The encoder used to encode the configs into the tensor.
        numerical_ord: The order of the norm to use for the numerical columns.
        categorical_ord: The order of the norm to use for the categorical columns.
        dtype: The dtype of the output tensor.
        square_form: If `True`, the output will be a square matrix of shape
            `(N, N)`. If `False`, the output will be a single dim tensor of shape
            `1/2 * N * (N - 1)`.

    Returns:
        The distances, shaped according to `square_form`.
    """
    categoricals = encoder.select_categorical(x)
    numericals = encoder.select_numerical(x)

    dists: torch.Tensor | None = None
    if numericals is not None:
        # Ensure they are all within the unit cube
        numericals = Domain.translate(
            numericals,
            frm=encoder.numerical_domains,
            to=UNIT_FLOAT_DOMAIN,
        )

        dists = torch.nn.functional.pdist(numericals, p=numerical_ord)

    if categoricals is not None:
        # Does Hamming distance
        cat_dists = torch.nn.functional.pdist(categoricals, p=categorical_ord)
        if dists is None:
            dists = cat_dists
        else:
            dists += cat_dists

    if dists is None:
        raise ValueError("No columns to compute distances on.")

    if not square_form:
        return dists

    # Turn the single dimensional vector into a square matrix
    N = len(x)
    sq = torch.zeros((N, N), dtype=dtype)
    row_ix, col_ix = torch.triu_indices(N, N, offset=1)
    sq[row_ix, col_ix] = dists
    sq[col_ix, row_ix] = dists
    return sq


def sample_one_old(
    space: SearchSpace,
    *,
    user_priors: bool = False,
    patience: int = 1,
    ignore_fidelity: bool = True,
) -> SearchSpace:
    """Sample a configuration from the search space.

    Args:
        space: The search space to sample from.
        user_priors: Whether to use user priors when sampling.
        patience: The number of times to try to sample a valid value for a
            hyperparameter.
        ignore_fidelity: Whether to ignore the fidelity parameter when sampling.

    Returns:
        A sampled configuration from the search space.
    """
    sampled_hps: dict[str, Parameter] = {}

    for name, hp in space.hyperparameters.items():
        if hp.is_fidelity and ignore_fidelity:
            sampled_hps[name] = hp.clone()
            continue

        for attempt in range(patience):
            try:
                if user_priors and isinstance(hp, ParameterWithPrior):
                    sampled_hps[name] = hp.sample(user_priors=user_priors)
                else:
                    sampled_hps[name] = hp.sample()
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Attempt {attempt + 1}/{patience} failed for"
                    f" sampling {name}: {e!s}"
                )
        else:
            logger.error(
                f"Failed to sample valid value for {name} after {patience} attempts"
            )
            raise ValueError(
                f"Could not sample valid value for hyperparameter {name}"
                f" in {patience} tries!"
            )

    return SearchSpace(**sampled_hps)
