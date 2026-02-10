from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Callable, Sequence

import torch

from neps.sampling import Prior, Sampler

if TYPE_CHECKING:
    from neps.space import ConfigEncoder
    from neps.space.parameters import Parameter


def make_initial_design(
    *,
    parameters: Mapping[str, Parameter],
    encoder: ConfigEncoder,
    sampler: Literal["sobol", "prior", "uniform"] | Sampler,
    sample_size: int | Literal["ndim"] | None = "ndim",
    sample_prior_first: bool = True,
    seed: torch.Generator | None = None,
    constraints_func: Callable[[Mapping[str, Any]], Sequence[float]] | None = None
) -> list[dict[str, Any]]:
    """Generate the initial design of the optimization process.

    Args:
        encoder: The encoder to use for encoding/decoding configurations.
        sampler: The sampler to use for the initial design.

            If set to "sobol", a Sobol sequence will be used.
            If set to "uniform", a uniform random sampler will be used.
            If set to "prior", a prior sampler will be used, based on the defaults,
                and confidence scores of the hyperparameters.
            If set to a custom sampler, the sampler will be used directly.

        sample_size:
            The number of configurations to sample.

            If "ndim", the number of configs will be equal to the number of dimensions.
            If None, no configurations will be sampled.

        sample_prior_first: Whether to sample the prior configuration first.
        seed: The seed to use for the random number generation.

    """
    configs: list[dict[str, Any]] = []
    if sample_prior_first:
        configs.append(
            {
                name: p.prior if p.prior is not None else p.center
                for name, p in parameters.items()
            }
        )

    ndims = len(parameters)
    if sample_size == "ndim":
        sample_size = ndims
    elif sample_size is not None and not sample_size > 0:
        raise ValueError(
            "The sample size should be a positive integer if passing an int."
        )

    if sample_size is not None:
        match sampler:
            case "sobol":
                sampler = Sampler.sobol(ndim=ndims)
            case "uniform":
                sampler = Sampler.uniform(ndim=ndims)
            case "prior":
                sampler = Prior.from_parameters(parameters)
            case _:
                pass

        encoded_configs = sampler.sample(sample_size * 2, to=encoder.domains, seed=seed)
        uniq_x = torch.unique(encoded_configs, dim=0)
        sample_configs = encoder.decode(uniq_x)
        if constraints_func is not None:
            valid_sample_configs = []
            for config in sample_configs:
                if constraints_func(config) >= 0:
                    valid_sample_configs.append(config)
                    if len(valid_sample_configs) >= sample_size:
                        break
            sample_configs = valid_sample_configs
        if constraints_func is not None and len(sample_configs) < sample_size:
                raise ValueError(
                    f"Could not sample enough valid configurations out of "
                    f"{sample_size * 2} samples, only got {len(sample_configs)} valid samples."
                )
        sample_configs = sample_configs[: sample_size]
        configs.extend(sample_configs)

    return configs
