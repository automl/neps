from functools import partial

from .multiscale_laplacian import MultiscaleLaplacian
from .vectorial_kernels import HammingKernel, Matern32Kernel, Matern52Kernel, RBFKernel
from .weisfilerlehman import WeisfilerLehman

StationaryKernelMapping = {
    "m52": Matern52Kernel(),
    "m32": Matern32Kernel(),
    "rbf": RBFKernel(),
    "hm": HammingKernel(),
}

GraphKernelMapping = {
    "wl": partial(
        WeisfilerLehman,
        h=2,
        oa=False,
    ),
    "mlk": partial(MultiscaleLaplacian, n=1),
    "vh": partial(
        WeisfilerLehman,
        h=0,
        oa=False,
    ),
}
