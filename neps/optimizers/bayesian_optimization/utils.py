import inspect
from collections import OrderedDict
from typing import Callable

import gpytorch
import torch

from .default_consts import LENGTHSCALE_SAFE_MARGIN

# Patch and fixes for gpytorch


class SafeInterval(gpytorch.constraints.Interval):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.safe_lower_bound = self.lower_bound + torch.tensor(LENGTHSCALE_SAFE_MARGIN)
        self.safe_upper_bound = self.upper_bound - torch.tensor(LENGTHSCALE_SAFE_MARGIN)

    def inverse_transform(self, transformed_tensor):
        transformed_tensor = torch.minimum(transformed_tensor, self.safe_upper_bound)
        transformed_tensor = torch.maximum(transformed_tensor, self.safe_lower_bound)
        return super().inverse_transform(transformed_tensor)


def get_default_args(func: Callable) -> dict:
    signature = inspect.signature(func)
    out_dict = OrderedDict()
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            out_dict[k] = v.default
    return out_dict


# Change the value of max_tries in cholesky
# assert gpytorch.utils.cholesky.psd_safe_cholesky.__defaults__ == (False, None, None, None)
_func_args_with_defaults = get_default_args(gpytorch.utils.cholesky.psd_safe_cholesky)
assert "max_tries" in _func_args_with_defaults
idx = list(_func_args_with_defaults.keys()).index("max_tries")
gpytorch.utils.cholesky.psd_safe_cholesky.__defaults__ = tuple(
    6 if idx == i else default
    for i, default in enumerate(gpytorch.utils.cholesky.psd_safe_cholesky.__defaults__)
)
