from __future__ import annotations

from copy import deepcopy

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
        rand_config = deepcopy(self.search_space)
        _patience = self.patience
        while _patience > 0:
            try:
                rand_config.sample()
                break
            except:  # pylint: disable=bare-except
                _patience -= 1
                continue
        if not _patience > 0:
            raise ValueError(
                f"Cannot sample valid random architecture in {self.patience} tries!"
            )
        return [rand_config]
