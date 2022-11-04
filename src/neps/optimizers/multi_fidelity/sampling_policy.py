from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from typing import Dict

from ...search_spaces.search_space import SearchSpace


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
    ):
        super().__init__(pipeline_space=pipeline_space)
        # setting all probabilities uniformly
        self.policy_map = {
            "random": 0.33,
            "prior": 0.34,
            "inc": 0.33
        }

    def sample(self, inc: SearchSpace, weights: Dict[str, float] = None) -> SearchSpace:
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
            # use inc to set the defaults of the configuration
            inc.set_defaults_to_current_values()
            # then sample with prior=True from that configuration
            # since the defaults are treated as the prior
            config = inc.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=True
            )
        else:
            print(f"Sampling from uniform with weights {prob_weights}")
            # random
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=True
            )
        return config
