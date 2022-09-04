from __future__ import annotations

from metahyper import instance_from_map

from ....search_spaces.architecture.core_graph_grammar import CoreGraphGrammar
from ....search_spaces.hyperparameters.categorical import CategoricalParameter
from ....search_spaces.hyperparameters.float import FloatParameter
from ....search_spaces.hyperparameters.integer import IntegerParameter
from ....utils.common import has_instance
from . import GraphKernelMapping, StationaryKernelMapping


def get_kernels(
    pipeline_space, domain_se_kernel, graph_kernels, hp_kernels, optimal_assignment
):
    if not graph_kernels:
        graph_kernels = []
        if has_instance(pipeline_space.values(), CoreGraphGrammar):
            graph_kernels.append("wl")
    if not hp_kernels:
        hp_kernels = []
        if has_instance(pipeline_space.values(), FloatParameter, IntegerParameter):
            hp_kernels.append("m52")
        if has_instance(pipeline_space.values(), CategoricalParameter):
            hp_kernels.append("hm")
    graph_kernels = [
        instance_from_map(GraphKernelMapping, kernel, "kernel", as_class=True)(
            oa=optimal_assignment,
            se_kernel=instance_from_map(
                StationaryKernelMapping, domain_se_kernel, "se kernel"
            ),
        )
        for kernel in graph_kernels
    ]
    hp_kernels = [
        instance_from_map(StationaryKernelMapping, kernel, "kernel")
        for kernel in hp_kernels
    ]
    if not graph_kernels and not hp_kernels:
        raise ValueError("No kernels are provided!")
    return graph_kernels, hp_kernels
