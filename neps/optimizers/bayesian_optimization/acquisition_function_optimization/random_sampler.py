from __future__ import annotations

from ....search_spaces.search_space import SearchSpace
from ..acquisition_functions.base_acquisition import BaseAcquisition
from .base_acq_optimizer import AcquisitionOptimizer


class RandomSampler(AcquisitionOptimizer):
    def __init__(
        self,
        search_space,
        acquisition_function: BaseAcquisition = None,
        patience: int = 100,
    ):
        super().__init__(search_space, acquisition_function)
        self.patience = patience

    def sample(self, pool_size: int, batch_size: None | int = None) -> list[SearchSpace]:
        rand_config = self.search_space.sample_new(patience=self.patience)
        return [rand_config]
