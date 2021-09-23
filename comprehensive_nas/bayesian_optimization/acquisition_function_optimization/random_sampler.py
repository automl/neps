from copy import deepcopy

from .base_acq_optimizer import AcquisitionOptimizer


class RandomSampler(AcquisitionOptimizer):
    def __init__(self, objective, patience: int = 100):
        super().__init__(objective=objective)
        self.patience = patience

    def sample(self, pool_size: int) -> list:
        pool = []
        while len(pool) < pool_size:
            rand_config = deepcopy(self.objective)
            _patience = self.patience
            while _patience > 0:
                try:
                    rand_config.sample_random_architecture()
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
