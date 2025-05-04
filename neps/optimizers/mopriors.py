from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from neps.sampling.priors import Prior
from neps.space import ConfigEncoder

if TYPE_CHECKING:
    from neps.space.parameters import Parameter


@dataclass
class MOPriorSampler:
    """A Sampler implementing a sampling strategy for multi-objective
    optimization with priors."""

    prior_dists: Mapping[str, Prior]

    parameters: Mapping[str, Parameter]

    encoder: ConfigEncoder

    @classmethod
    def from_mapping(
        cls,
        parameters: Mapping[str, Parameter],
        prior_centers: Mapping[str, Mapping[str, float]],
        confidence_values: Mapping[str, Mapping[str, float]],
    ) -> MOPriorSampler:
        """Creates a MOPriorSampler instance.

        Args:
            parameters: The parameters to sample from.
            prior_centers: The priors to use for sampling.
            confidence_values: The confidence values for the priors.

        Returns:
            The MOPriorSampler instance.
        """
        _priors = {}

        for key, _prior_center in prior_centers.items():
            assert isinstance(_prior_center, dict), (
                f"Expected prior center values to be a dict, got {type(_prior_center)}"
            )
            assert key in confidence_values, (
                f"Expected confidence values to contain {key}, "
                f"got {confidence_values.keys()}"
            )
            _priors[key] = Prior.from_parameters(
                parameters=parameters,
                center_values=_prior_center,
                confidence_values=confidence_values[key],
            )

        return MOPriorSampler(
            prior_dists=_priors,
            parameters=parameters,
            encoder=ConfigEncoder.from_parameters(parameters),
        )

    def sample_config(self) -> dict[str, Any]:
        """Samples a configuration using the MOPriors algorithm.

        Returns:
            The sampled configuration.
        """
        _prior_choice: Prior = np.random.choice(
            list(self.prior_dists.values()),
        )

        # Sample a configuration from the chosen prior
        return _prior_choice.sample_config(to=self.encoder)
