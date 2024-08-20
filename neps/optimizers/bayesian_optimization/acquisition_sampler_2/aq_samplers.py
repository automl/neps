from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from neps.search_spaces import SearchSpace


def random_sample(search_space: SearchSpace, *, seed: torch.Generator) -> SearchSpace:
    """Sample a random value from a search space.

    Args:
        search_space: The search space to sample from.
        user_priors: Whether to sample from user priors.
        seed: The seed to use for sampling.

    Returns:
        A search space with a sampled value.
    """
    return search_space.sample_value(user_priors=user_priors)
