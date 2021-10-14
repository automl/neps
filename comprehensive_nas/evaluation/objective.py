import time
from abc import abstractmethod

import numpy as np


class Objective:
    def __init__(self, seed: int, log_scale: bool, negative: bool = False) -> None:
        self.seed = seed
        self.log_scale = log_scale
        self.negative = negative

    @abstractmethod
    def __call__(self, config, mode, **kwargs):
        raise NotImplementedError

    def set_seed(self, seed: int):
        self.seed = seed

    def transform(self, val):
        if self.log_scale:
            val = np.log(val + 1e-8)  # avoid log(0)
        if self.negative:
            val *= -1
        return val

    def inv_transform(self, val):
        if self.negative:
            val *= -1
        if self.log_scale:
            val = np.exp(val)
        return val


class ObjectiveWithAPI(Objective):
    def __init__(self, api) -> None:
        super().__init__(None, None, None)
        self.api = api

    def __call__(self, config):
        _config = config.get_model_for_evaluation()
        start = time.time()
        loss = self.api.eval(_config)
        end = time.time()
        return {
            "loss": loss,
            "info_dict": {
                "config_id": config.id,
                "val_score": self.api.inv_transform(loss),
                "test_score": self.api.inv_transform(self.api.test(_config)),
                "train_time": end - start,
            },
        }
