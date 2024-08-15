from __future__ import annotations

from neps.optimizers.bayesian_optimization.kernels import Kernel
from neps.optimizers.bayesian_optimization.kernels.vectorial_kernels import (
    HammingKernel,
    Matern52Kernel,
)
import torch
from neps.optimizers.bayesian_optimization.kernels.weisfilerlehman import WeisfilerLehman

from neps.search_spaces import SearchSpace


# TODO: Option to combine numerical and categorical into one.
def get_default_kernels(
    *,
    space: SearchSpace,
    optimizable: bool = True,
) -> list[tuple[Kernel, list[str]],]:
    kernels: list[tuple[Kernel, list[str]]] = []
    if any(space.graphs):
        h = 2
        if optimizable:
            layer_weights = torch.nn.Parameter(torch.ones(h + 1))
        else:
            layer_weights = None

        kernels.append(
            (
                WeisfilerLehman(h=2, layer_weights=layer_weights, oa=True),
                list(space.graphs.keys()),
            )
        )

    if any(space.categoricals):
        if optimizable:
            lengthscales = torch.nn.Parameter(torch.ones(len(space.categoricals)))
        else:
            lengthscales = torch.ones(len(space.categoricals))

            kernels.append(
                (
                    HammingKernel(lengthscale=lengthscales),
                    list(space.categoricals.keys()),
                )
            )

    if any(space.numerical):
        if optimizable:
            lengthscales = torch.nn.Parameter(torch.ones(len(space.numerical)))
        else:
            lengthscales = torch.ones(len(space.numerical))

            kernels.append(
                (Matern52Kernel(lengthscale=lengthscales), list(space.numerical.keys()))
            )

    return kernels
