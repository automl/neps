import random
from typing import Tuple

from ..core.optimizer import Optimizer


class RandomSearch(Optimizer):
    def __init__(self, acquisition_function_opt=None, return_opt_details: bool = False):
        super().__init__()
        self.sampled_idx = []
        self.acquisition_function_opt = acquisition_function_opt
        self.return_opt_details = return_opt_details
        self.surrogate_model = None

    def initialize_model(self, **kwargs):
        pass

    def update_model(self, **kwargs):
        pass

    def propose_new_location(
        self, batch_size: int = 5, n_candidates: int = 10
    ) -> Tuple[Tuple, dict]:
        # create candidate pool
        pool = self.acquisition_function_opt.sample(n_candidates)

        next_x = random.sample(pool, batch_size)
        self.sampled_idx.append(next_x)

        opt_details = {"pool": pool} if self.return_opt_details else None

        return next_x, opt_details
