from functools import partial

from .abstract_benchmark import *
from .hpo.branin2 import Branin2
from .hpo.counting_ones import CountingOnes
from .hpo.hartmann3 import Hartmann3
from .hpo.hartmann6 import Hartmann6
from .nas.nasbench201 import NASBench201
from .nas.nasbench301 import NASBench301

BenchmarkMapping = {
    "branin2": partial(Branin2, negative=True),
    "hartmann3": partial(Hartmann3, negative=False),
    "hartmann6": partial(Hartmann6, negative=False),
    "counting_ones": partial(CountingOnes, negative=False),
    # TODO: fix this path
    "nasbench201": partial(
        NASBench201,
        data_dir="comprehensive_nas/bo/" "benchmarks/nas/nb_configfiles/data/",
        negative=True,
    ),
    "nasbench301": partial(NASBench301, negative=True),
}
