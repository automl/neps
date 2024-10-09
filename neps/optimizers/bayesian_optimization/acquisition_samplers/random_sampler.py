from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from neps.optimizers.bayesian_optimization.acquisition_samplers import AcquisitionSampler

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace


# TODO: Chop this the hell out, it's pretty bad
# We have much better and efficient ways to generate acquisition samples now
class RandomSampler(AcquisitionSampler):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        patience: int = 100,
        budget: int | None = None,  # TODO: Remove
    ):
        super().__init__(pipeline_space=pipeline_space, patience=patience)

    def sample(self, acquisition_function: Callable | None = None) -> SearchSpace:
        return self.pipeline_space.sample(
            patience=self.patience, user_priors=False, ignore_fidelity=False
        )
