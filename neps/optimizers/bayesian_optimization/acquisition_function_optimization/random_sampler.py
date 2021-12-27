from copy import deepcopy
from typing import Tuple

import numpy as np

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

    def create_pool(self, pool_size: int) -> list:
        pool = []
        while len(pool) < pool_size:
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
            pool.append(rand_config)
        return pool

    def sample(
        self, pool_size: int, batch_size: int = None
    ) -> Tuple[list, list, np.ndarray]:
        pool = self.create_pool(pool_size)

        if batch_size is None:
            return pool

        if batch_size is not None and self.acquisition_function is None:
            raise Exception("Random sampler has no acquisition function!")

        samples, acq_vals, _ = self.acquisition_function.propose_location(
            top_n=batch_size, candidates=pool
        )
        return samples, pool, acq_vals
