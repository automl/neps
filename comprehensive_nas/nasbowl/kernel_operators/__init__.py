from .weisfilerlehman import *
from .combine_kernels import *
from .vectorial_kernels import *
from .multiscale_laplacian import MultiscaleLaplacian
from .other_kernels import *
from .encoding import *

from functools import partial

StationaryKernelMapping = {
    'm52': Matern52Kernel(),
    'm32': Matern32Kernel(),
    'rbf': RBFKernel(),
    'hm': HammingKernel()
}

GraphKernelMapping = {
    'wl': partial(WeisfilerLehman, h=2, oa=False, ),
    'mlk': partial(MultiscaleLaplacian, n=1),
    'vh': partial(WeisfilerLehman, h=0, oa=False, ),
}