from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence
from typing_extensions import Self

import torch
import torch.nn as nn
from itertools import product

import numpy as np
from neps.optimizers.bayesian_optimization.kernels.grakel_replace import (
    VertexHistogram,
    WeisfeilerLehman as _WL,
)
from neps.optimizers.bayesian_optimization.kernels.kernel import Kernel
from neps.optimizers.bayesian_optimization.kernels.vectorial_kernels import Stationary
from neps.search_spaces.encoding import WLInput

GRID_WL_LENGTHSCALES = torch.tensor([np.e**i for i in range(-2, 3)])
GRID_WL_SUBTREE_CANDIDATES = (1, 2, 3, 4, 5)


class WeisfilerLehman(Kernel[Sequence[WLInput]]):
    """Weisfiler Lehman kernel using grakel functions."""

    suggested_grid: ClassVar[Sequence[Mapping[str, Any]]] = [
        {"h": h, "se_kernel": Stationary(lengthscale=l)}
        for h, l in product(GRID_WL_SUBTREE_CANDIDATES, GRID_WL_LENGTHSCALES)
    ]

    def __init__(
        self,
        h: int = 0,
        se_kernel: Stationary | None = None,
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
                If None, uses uniform
            oa: whether the optimal assignment variant of the Weisfiler-Lehman
                kernel should be used
            node_label: the node_label defining the key node attribute.
        """
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

    def fit_transform(self, gr: Sequence[WLInput]) -> torch.Tensor:
        self.layer_weights.clamp_(0, 1)
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
            layer_weights=self.layer_weights,
            normalize=True,
        )

        K = self.wl_kernel_.fit_transform(gr)
        return torch.as_tensor(K, dtype=torch.float64)

    def transform(self, gr: Sequence[WLInput]) -> torch.Tensor:
        assert self.wl_kernel_ is not None
        self.layer_weights.clamp_(0, 1)

        K = self.wl_kernel_.transform(gr)
        return torch.as_tensor(K, dtype=torch.float64)
