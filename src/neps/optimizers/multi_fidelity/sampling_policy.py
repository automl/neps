from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity_prior.utils import compute_config_dist

SAMPLE_THRESHOLD = 1000
MAX_PERCENTILE = 25
DELTA_PERCENTILE = 0.1


class SamplingPolicy(ABC):
    """Base class for implementing a sampling straregy for SH and its subclasses"""

    def __init__(self, pipeline_space: SearchSpace, patience: int = 100):
        self.pipeline_space = pipeline_space
        self.patience = patience

    @abstractmethod
    def sample(self) -> SearchSpace:
        pass


class RandomUniformPolicy(SamplingPolicy):
    """A random policy for sampling configuration, i.e. the default for SH / hyperband

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
    ):
        super().__init__(pipeline_space=pipeline_space)

    def sample(self) -> SearchSpace:
        return self.pipeline_space.sample(
            patience=self.patience, user_priors=False, ignore_fidelity=True
        )


class FixedPriorPolicy(SamplingPolicy):
    """A random policy for sampling configuration, i.e. the default for SH but samples
    a fixed fraction from the prior.
    """

    def __init__(self, pipeline_space: SearchSpace, fraction_from_prior: float = 1):
        super().__init__(pipeline_space=pipeline_space)
        assert 0 <= fraction_from_prior <= 1
        self.fraction_from_prior = fraction_from_prior

    def sample(self) -> SearchSpace:
        """Samples from the prior with a certain probabiliyu

        Returns:
            SearchSpace: [description]
        """
        user_priors = False
        if np.random.uniform() < self.fraction_from_prior:
            user_priors = True
        config = self.pipeline_space.sample(
            patience=self.patience, user_priors=user_priors, ignore_fidelity=True
        )
        return config


class EnsemblePolicy(SamplingPolicy):
    """Ensemble of sampling policies including sampling randomly, from prior & incumbent.

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        dist_type: str = "hypersphere",
    ):
        """Samples a policy as per its weights and performs the selected sampling.

        Args:
            pipeline_space: Space in which to search
            dist_type: str
                if "hypersphere", uniformly samples from around the incumbent within its
                distance from the nearest neighbour in history
                if "gaussian", samples from a gaussian around the incumbent
        """
        super().__init__(pipeline_space=pipeline_space)
        self.dist_type = dist_type
        # setting all probabilities uniformly
        self.policy_map = {"random": 0.33, "prior": 0.34, "inc": 0.33}

    def sample_neighbour(self, incumbent, distances, tolerance=1e-4):
        """Samples a config from around the `incumbent` within radius as `distance`."""
        # TODO: how does tolerance affect optimization on landscapes of different scale
        percentile = 0
        sample_counter = 0
        while True:
            # sampling a config
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=False
            )
            # computing distance from incumbent
            d = compute_config_dist(config, incumbent)
            # thresholding hypersphere radius
            distance = np.percentile(distances, percentile)
            if d < max(distance, tolerance):
                # accept sample
                break
            sample_counter += 1
            if sample_counter > SAMPLE_THRESHOLD:
                sample_counter = 0
                # if no sample falls within the radius, increase the threshold based
                # on the set of distances seen, capping the threshold to 25% of all
                # distances from the incumbent, loosely translates to the radius being
                # the maximum of the 25-th percentile of distanes from the incumbent
                percentile = min(percentile + DELTA_PERCENTILE, MAX_PERCENTILE)
        # end of while
        return config

    def sample(
        self, inc: SearchSpace, weights: dict[str, float] = None, **kwargs
    ) -> SearchSpace:
        """Samples from the prior with a certain probabiliyu

        Returns:
            SearchSpace: [description]
        """
        if weights is not None:
            for key, value in sorted(weights.items()):
                self.policy_map[key] = value
        assert sum(self.policy_map.values()) == 1, "Policy prob. weights should sum to 1."
        prob_weights = [v for _, v in sorted(self.policy_map.items())]
        policy_idx = np.random.choice(range(len(prob_weights)), p=prob_weights)
        policy = sorted(self.policy_map.keys())[policy_idx]

        if policy == "prior":
            print(f"Sampling from prior with weights {prob_weights}")
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=True
            )
        elif policy == "inc":
            print(f"Sampling from inc with weights {prob_weights}")

            if self.dist_type == "hypersphere":
                distance = kwargs["distance"]
                config = self.sample_neighbour(inc, distance)
            elif self.dist_type == "gaussian":
                # use inc to set the defaults of the configuration
                inc.set_defaults_to_current_values()
                # then sample with prior=True from that configuration
                # since the defaults are treated as the prior
                config = inc.sample(
                    patience=self.patience, user_priors=True, ignore_fidelity=True
                )
            else:
                raise ValueError(
                    f"{self.dist_type} is not in {{'hypersphere', 'gaussian'}}"
                )
        else:
            print(f"Sampling from uniform with weights {prob_weights}")
            # random
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=True
            )
        return config
