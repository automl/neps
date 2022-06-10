import gpytorch
import torch


class SafeInterval(gpytorch.constraints.Interval):
    def inverse_transform(self, transformed_tensor):
        transformed_tensor = torch.minimum(transformed_tensor, self.upper_bound)
        transformed_tensor = torch.maximum(transformed_tensor, self.lower_bound)
        return super().inverse_transform(transformed_tensor)
