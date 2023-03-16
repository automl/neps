from __future__ import annotations

import logging

import torch


_logger = logging.getLogger(__name__)


# Define the combining kernel

class CombiningKernel(torch.nn.Module):
    def __init__(
        self,
        n_kernels: int,
        combining_kernel_variant: str,
    ):
        super().__init__()

        if n_kernels < 1:
            raise ValueError(f"n_kernels can not be < 1: {n_kernels}")
        self.n_kernels = n_kernels

        if combining_kernel_variant not in ("sum", "product"):
            raise ValueError(
                "Invalid value for combining kernel variant: "
                + f"{combining_kernel_variant!r}"
            )
        self.combining_kernel_variant = combining_kernel_variant

        self.combining_weights = torch.nn.Parameter(
            torch.tensor([1.0 / n_kernels for _ in range(n_kernels)]),
            requires_grad=True,
        )

    def forward(self, kernel_results: torch.Tensor) -> torch.Tensor:
        kernel_results = kernel_results.clone()

        # We expect kernel_results to have size (n_kernels, n_configs, n_configs)
        kernel_results_size = kernel_results.size()

        if len(kernel_results_size) != 3:
            raise ValueError(
                "kernel_results does not have 3 dimensions as it should: "
                + f"{len(kernel_results_size)} != 3"
            )
        if kernel_results_size[0] != self.n_kernels:
            raise ValueError(
                "Size of kernel_results != n_kernels: "
                + f"{kernel_results_size[0]} != {self.n_kernels}"
            )
        if kernel_results_size[1] != kernel_results_size[2]:
            raise ValueError(
                "For each kernel the result should be a square matrix: "
                + f"{kernel_results_size[1]} != {kernel_results_size[2]}"
            )

        # # Adjust the weights to be in a good range
        combining_weights = torch.softmax(self.combining_weights, dim=0)

        # Multiply kernel results by their weights
        kernel_results_weighted = combining_weights.view(-1, 1, 1) * kernel_results
        assert kernel_results_weighted.size() == kernel_results_size

        # Do the combining operation
        if self.combining_kernel_variant == "sum":
            K = torch.sum(kernel_results_weighted, dim=0)
        elif self.combining_kernel_variant == "product":
            K = torch.prod(kernel_results_weighted, dim=0)
        else:
            raise ValueError(
                "Invalid value for combining kernel variant: "
                + f"{self.combining_kernel_variant!r}"
            )
        assert K.size() == (kernel_results_size[1], kernel_results_size[2])

        return K
