from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

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
        return self.pipeline_space.sample(
            patience=self.patience, user_priors=user_priors, ignore_fidelity=True
        )
