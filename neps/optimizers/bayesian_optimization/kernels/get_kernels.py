from __future__ import annotations

from neps.search_spaces import (
    FloatParameter,
    IntegerParameter,
    CategoricalParameter,
    CoreGraphGrammar,
)
from neps.utils.common import has_instance, instance_from_map
from neps.optimizers.bayesian_optimization.kernels import (
    GraphKernelMapping,
    StationaryKernelMapping,
)


def get_kernels(
    pipeline_space,
    domain_se_kernel,
    graph_kernels,
    hp_kernels,
    optimal_assignment,
):
    params = list(pipeline_space.hyperparameters.values())
    if not graph_kernels:
        graph_kernels = []
        if has_instance(params, CoreGraphGrammar):
            graph_kernels.append("wl")

    graph_kernels = [
        instance_from_map(GraphKernelMapping, kernel, "kernel", as_class=True)(
            oa=optimal_assignment,
            se_kernel=instance_from_map(
                StationaryKernelMapping, domain_se_kernel, "se kernel"
            ),
        )
        for kernel in graph_kernels
    ]

    if not hp_kernels:
        hp_kernels = []
        if has_instance(params, FloatParameter, IntegerParameter):
            hp_kernels.append("m52")
        if has_instance(params, CategoricalParameter):
            hp_kernels.append("hm")

    hp_kernels = [
        instance_from_map(StationaryKernelMapping, kernel, "kernel")
        for kernel in hp_kernels
    ]

    if not graph_kernels and not hp_kernels:
        raise ValueError("No kernels are provided!")

    return graph_kernels, hp_kernels
