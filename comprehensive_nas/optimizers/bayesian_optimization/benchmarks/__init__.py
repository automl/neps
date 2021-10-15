from functools import partial

from .abstract_benchmark import *
from .hpo.branin2 import Branin2, evaluate_branin2
from .hpo.counting_ones import CountingOnes, evaluate_counting_ones
from .hpo.hartmann3 import Hartmann3, evaluate_hartmann3
from .hpo.hartmann6 import Hartmann6, evaluate_hartmann6
from .nas.nasbench201 import NASBench201
from .nas.nasbench301 import NASBench301

BenchmarkMapping = {
    "branin2": Branin2,
    "hartmann3": Hartmann3,
    "hartmann6": Hartmann6,
    "counting_ones": CountingOnes,
    # TODO: fix this path
    "nasbench201": partial(
        NASBench201,
        negative=True,
    ),
    "nasbench301": partial(NASBench301, negative=True),
}

PipelineFunctionMapping = {
    "branin2": evaluate_branin2,
    "hartmann3": evaluate_hartmann3,
    "hartmann6": evaluate_hartmann6,
    "counting_ones": evaluate_counting_ones,
    # "nasbench301": None,
}
