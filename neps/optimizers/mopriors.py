from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from neps.sampling.priors import Prior

if TYPE_CHECKING:
    from neps.space import ConfigEncoder
    from neps.space.parameters import Parameter


@dataclass
class MOPriorSampler:
    """A Sampler implementing a sampling strategy for multi-objective
    optimization with priors."""

    prior_dists: Mapping[str, Prior]

    parameters: Mapping[str, Parameter]

    encoder: ConfigEncoder

    @classmethod
    def dists_from_centers_and_confidences(
        cls,
        parameters: Mapping[str, Parameter],
        prior_centers: Mapping[str, Mapping[str, float]],
        confidence_values: Mapping[str, Mapping[str, float]] | None = None,
    ) -> Mapping[str, Prior]:
        """Creates a mapping of prior distributions from the given centers and
        confidence values.
        Args:
            parameters: The parameters to sample from.
            prior_centers: The priors to use for sampling.
            confidence_values: The confidence values for the priors.
        Returns:
            A mapping of prior distributions.
        """
        _priors = {}
        for key, _prior_center in prior_centers.items():
            assert isinstance(_prior_center, dict), (
                f"Expected prior center values to be a dict, got {type(_prior_center)}"
            )
            _default_confidence = dict.fromkeys(prior_centers.keys(), 0.25)
            _priors[key] = Prior.from_parameters(
                parameters=parameters,
                center_values=_prior_center,
                confidence_values=(
                    confidence_values[key] if confidence_values else _default_confidence
                ),
            )
        return _priors

    @classmethod
    def create_sampler(
        cls,
        parameters: Mapping[str, Parameter],
        prior_centers: Mapping[str, Mapping[str, float]],
        confidence_values: Mapping[str, Mapping[str, float]],
        encoder: ConfigEncoder,
    ) -> MOPriorSampler:
        """Creates a MOPriorSampler instance.

        Args:
            parameters: The parameters to sample from.
            prior_centers: The priors to use for sampling.
            confidence_values: The confidence values for the priors.
            encoder: The encoder to use for encoding and decoding configurations
                into tensors.

        Returns:
            The MOPriorSampler instance.
        """
        _priors = cls.dists_from_centers_and_confidences(
            parameters=parameters,
            prior_centers=prior_centers,
            confidence_values=confidence_values,
        )

        return MOPriorSampler(
            prior_dists=_priors,
            parameters=parameters,
            encoder=encoder,
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
