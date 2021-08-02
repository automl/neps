from abc import abstractmethod


class AcquisitionOptimizer:
    def __init__(self, objective):
        self.objective = objective
        self.x = None
        self.y = None

    @abstractmethod
    def sample(self, pool_size: int):
        raise NotImplementedError

    def reset_XY(self, x, y):
        self.x = x
        self.y = y
