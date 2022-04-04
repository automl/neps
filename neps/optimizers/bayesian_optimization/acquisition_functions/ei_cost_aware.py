from typing import Iterable, Union
import numpy as np
import torch

from .ei import ComprehensiveExpectedImprovement

class CArBO(ComprehensiveExpectedImprovement):
    def __init__(
        self,
        surrogate_model,
        augmented_ei: bool = False,
        xi: float = 0.0,
        in_fill: str = "best",
        log_ei: bool = False,
        compute_fast: bool = True,
    ):
        assert(not augmented_ei)
        super().__init__(surrogate_model, augmented_ei, xi, in_fill, log_ei, compute_fast)

    def eval(
            self, x: Iterable, alpha : float, asscalar: bool = False
    ) -> Union[np.ndarray, torch.Tensor, float]:
        ei = super().eval(x, asscalar)
        # TODO get costs
        costs = 1
        return ei / (costs ^ alpha)
