from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int = 100):
    """Seed all random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure reproducibility with CuDNN (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def min_max_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Scale the input tensor to the range [0, 1]."""
    min_vals = tensor.min(dim=0, keepdim=True).values
    max_vals = tensor.max(dim=0, keepdim=True).values
    return (tensor - min_vals) / (max_vals - min_vals)
