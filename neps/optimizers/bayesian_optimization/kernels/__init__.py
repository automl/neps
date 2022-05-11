from __future__ import annotations

from functools import partial
from typing import Callable

from .base_kernel import Kernel
from .combine_kernels import CombineKernel, ProductKernel, SumKernel
from .vectorial_kernels import CategoricalBotorchKernel, HammingKernel, MaternKernel

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
    # "rbf": RBFKernel,
    "categorical": CategoricalBotorchKernel,
    "hm": HammingKernel,
    **CombineKernelMapping,
}
