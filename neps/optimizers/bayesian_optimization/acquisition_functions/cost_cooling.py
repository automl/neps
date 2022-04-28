from typing import Iterable, Union

import numpy as np
import torch

from .base_acquisition import BaseAcquisition
from .ei import ComprehensiveExpectedImprovement


class CostCooling(BaseAcquisition):
    def __init__(
        self,
        base_acquisition: BaseAcquisition,
        # TODO(Jan): Args
    ):  # pylint: disable=super-init-not-called
        pass  # TODO(Jan)

    def eval(
        self,
        x: Iterable,
        **base_acquisition_kwargs,
    ) -> Union[np.ndarray, torch.Tensor, float]:
        base_acquisition_value = self.base_acquisition(x, **base_acquisition_kwargs)
        # TODO get costs
        costs = 1
        return base_acquisition_value / (costs ^ alpha)
