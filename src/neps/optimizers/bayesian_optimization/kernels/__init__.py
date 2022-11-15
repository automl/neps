from __future__ import annotations

from functools import partial
from typing import Callable

from ....search_spaces.hyperparameters.constant import (
    ConstantParameter as _ConstantParameter,
)
from .base_kernel import CustomKernel, Kernel, NoOpKernel
from .combine_kernels import CombineKernel, ProductKernel, SumKernel
from .get_kernels import instantiate_kernel
from .graph_kernel import GraphKernel
from .vectorial_kernels import (
    BaseCategoricalKernel,
    BaseNumericalKernel,
    CategoricalBotorchKernel,
    HammingKernel,
    LayeredRBFKernel,
    MaternKernel,
    RationalQuadraticKernel,
    RBFKernel,
)
from .weisfeilerlehman import WeisfeilerLehman  # type: ignore[attr-defined]

GraphKernelMapping: dict[str, Callable] = {
    "wl": partial(
        WeisfeilerLehman,
        h=2,
        oa=False,
    ),
    "vh": partial(
        WeisfeilerLehman,
        h=0,
        oa=False,
    ),
}

CombineKernelMapping: dict[str, Callable] = {
    "sum": SumKernel,
    "product": ProductKernel,
}

KernelMapping: dict[str, Callable] = {
    "m32": partial(MaternKernel, nu=3 / 2),
    "m52": partial(MaternKernel, nu=5 / 2),
    "rbf": RBFKernel,
    "categorical": CategoricalBotorchKernel,
    "hm": HammingKernel,
    "no_op": NoOpKernel,
    "no_op_const": partial(NoOpKernel, restrict_to=[_ConstantParameter]),
    **GraphKernelMapping,
    **CombineKernelMapping,
}
