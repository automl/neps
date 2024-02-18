from __future__ import annotations

from functools import partial
from typing import Callable

from .encoding import NASBOTDistance
from .vectorial_kernels import HammingKernel, Matern32Kernel, Matern52Kernel, RBFKernel
from .weisfilerlehman import WeisfilerLehman

StationaryKernelMapping: dict[str, Callable] = {
    "m52": Matern52Kernel,
    "m32": Matern32Kernel,
    "rbf": RBFKernel,
    "hm": HammingKernel,
}

GraphKernelMapping: dict[str, Callable] = {
    "wl": partial(
        WeisfilerLehman,
        h=2,
        oa=False,
    ),
    "vh": partial(
        WeisfilerLehman,
        h=0,
        oa=False,
    ),
    "nasbot": NASBOTDistance,
}
