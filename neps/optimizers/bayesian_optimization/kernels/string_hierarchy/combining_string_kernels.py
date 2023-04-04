from __future__ import annotations

import logging

import torch

_logger = logging.getLogger(__name__)


# CombiningKernel

class CombiningKernel(torch.nn.Module):
    def __init__(
        self,
        n_kernels: int,
        combining_kernel_variant: str,
    ):
        super().__init__()

        self.n_kernels = n_kernels
        self.combining_kernel_variant = combining_kernel_variant

        self.combining_weights = torch.nn.Parameter(
            torch.tensor([1.0 / n_kernels for _ in range(n_kernels)]),
            requires_grad=True,
        )

    def forward(self, kernel_results: torch.Tensor) -> torch.Tensor:
        _logger.debug(f"Called method `forward` of kernel `%s`", self)
        _logger.debug("Combining weights: %s", self.combining_weights)

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

        # Adjust the combining weights to an appropriate range
        # Weights were initialized with equal values, so their values
        #  after softmax will also initially be equal between them.
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

        assert bool((torch.diag(K) != 0.0).all()), f"Found value 0.0 in diagonal: {K}"
        assert bool((torch.max(K, dim=1).values - torch.diag(K) <= 1e-5).all()), (
            f"Max value not in diagonal: {torch.max(K, dim=1).values}, {torch.diag(K)}, "
            f"{torch.max(K, dim=1).values == torch.diag(K)}"
        )

        return K
