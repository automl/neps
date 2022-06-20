from __future__ import annotations

from functools import partial
from typing import Callable

from ....search_spaces.numerical.constant import ConstantParameter as _ConstantParameter
from .base_kernel import CustomKernel, Kernel, NoOpKernel
from .combine_kernels import CombineKernel, ProductKernel, SumKernel
from .get_kernels import instantiate_kernel
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

# GraphKernelMapping: dict[str, Callable] = {
#     "wl": partial(
#         WeisfilerLehman,
#         h=2,
#         oa=False,
#     ),
#     "vh": partial(
#         WeisfilerLehman,
#         h=0,
#         oa=False,
#     ),
# }

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
    **CombineKernelMapping,
}
