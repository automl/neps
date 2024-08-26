from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from neps.optimizers.bayesian_optimization.kernels.grakel_replace import (
    VertexHistogram,
    WeisfeilerLehman as _WL,
)
from neps.optimizers.bayesian_optimization.kernels.kernel import Kernel

if TYPE_CHECKING:
    from neps.optimizers.bayesian_optimization.kernels.vectorial_kernels import (
        NumericKernel,
    )

GRID_WL_LENGTHSCALES = torch.tensor([np.e**i for i in range(-2, 3)])
GRID_WL_SUBTREE_CANDIDATES = (1, 2, 3, 4, 5)


def normal_prior(param: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return -0.5 * torch.sum(((param - mean) / std) ** 2) - torch.sum(
        torch.log(std * torch.sqrt(2 * torch.tensor(np.pi)))
    )


def kernel_hp_prior(params: dict[str, nn.Parameter]) -> torch.Tensor:
    return normal_prior(params["layer_weights"], mean=0, std=1)


class WeisfilerLehman(Kernel[npt.NDArray[np.object_]]):
    """Weisfiler Lehman kernel using grakel functions."""

    def __init__(
        self,
        *,
        h: int = 0,
        se_kernel: NumericKernel | None = None,
        layer_weights: torch.Tensor | None = None,
        oa: bool = False,
        node_label: str = "op_name",
    ):
        """Initializes the Weisfeiler-Lehman kernel.

        Args:
            h: The number of Weisfeiler-Lehman iterations
            se_kernel: defines a stationary vector kernel to be used for
                successive embedding (i.e. the kernel function on which the
                vector embedding inner products are computed).
                If None, uses the default linear kernel
            layer_weights: The weights for each layer of the Weisfeiler-Lehman kernel.
                If None, uses uniform 1s
            oa: whether the optimal assignment variant of the Weisfiler-Lehman
                kernel should be used
            node_label: the node_label defining the key node attribute.
        """
        super().__init__(hyperparameter_prior=kernel_hp_prior)
        if se_kernel is not None and oa:
            raise ValueError(
                "Only one or none of se (successive embedding) and oa (optimal assignment) may be true!"
            )

        self.h = h
        self.se_kernel = se_kernel
        self.layer_weights = (
            layer_weights if layer_weights is not None else torch.ones(h + 1)
        )
        self.oa = oa
        self.node_label = node_label
        if node_label != "op_name":
            raise NotImplementedError("Only 'op_name' is supported for node_label")

        self.wl_kernel_: _WL | None = None

    def as_optimizable(self) -> Self:
        return self.clone_with(layer_weights=nn.Parameter(self.layer_weights))

    def fit_transform(self, gr: npt.NDArray[np.object_]) -> torch.Tensor:
        self.wl_kernel_ = _WL(
            h=self.h,
            base_graph_kernel=(  # type: ignore
                VertexHistogram,
                {
                    "sparse": False,
                    "se_kernel": self.se_kernel,
                    "oa": self.oa,
                    "requires_ordered_features": True,
                },
            ),
            layer_weights=self.layer_weights / self.layer_weights.sum(),
            normalize=True,
        )

        K = self.wl_kernel_.fit_transform(iter(gr))
        return torch.as_tensor(K, dtype=torch.float64)

    def transform(self, gr: npt.NDArray[np.object_]) -> torch.Tensor:
        assert self.wl_kernel_ is not None

        K = self.wl_kernel_.transform(iter(gr))
        return torch.as_tensor(K, dtype=torch.float64)

    def forward(
        self,
        x: npt.NDArray[np.object_],
        x2: npt.NDArray[np.object_] | None = None,
    ) -> torch.Tensor:
        if x2 is None:
            K = self.fit_transform(x)
            self.wl_kernel_ = None
            return K

        self.fit_transform(x)
        K = self.transform(x2)
        self.wl_kernel_ = None
        return K
