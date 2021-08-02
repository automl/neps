import random
from typing import List, Tuple

from ..core.optimizer import Optimizer


class RandomSearch(Optimizer):
    def __init__(self, acquisition_function_opt=None):
        super().__init__()
        self.sampled_idx = []
        self.acquisition_function_opt = acquisition_function_opt
        self.surrogate_model = None

    def initialize_model(self, **kwargs):
        pass

    def update_model(self, **kwargs):
        pass

    def propose_new_location(
        self, batch_size: int = 5, pool_size: int = 10
    ) -> Tuple[List, List[float]]:
        # create candidate pool
        pool = self.acquisition_function_opt.sample(pool_size)

        next_x = random.sample(pool, batch_size)
        self.sampled_idx.append(next_x)

        return next_x, pool
