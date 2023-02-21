import torch

_TOLERANCE = 1E-6


def tensor_is_symmetric(tensor: torch.Tensor):
    return bool((torch.abs(tensor - tensor.T) <= _TOLERANCE).all())


def tensor_is_positive_semi_definite(tensor: torch.Tensor):
    return (
        tensor_is_symmetric(tensor)
        and bool((torch.linalg.eigvalsh(tensor) >= 0).all())
    )
