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


# Change the value of max_tries in cholesky
assert gpytorch.utils.cholesky.psd_safe_cholesky.__defaults__ == (False, None, None, 3)
gpytorch.utils.cholesky.psd_safe_cholesky.__defaults__ = (False, None, None, 6)
