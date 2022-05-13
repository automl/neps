from __future__ import annotations

from metahyper import instance_from_map

import neps

from ....search_spaces.graph_grammar.core_graph_grammar import CoreGraphGrammar
from ....utils.common import has_instance
from ..default_consts import DEFAULT_KERNELS


def get_kernels(
    pipeline_space, domain_se_kernel, graph_kernels, hp_kernels, optimal_assignment
):
    kernel_map = neps.optimizers.bayesian_optimization.kernels.KernelMapping
    if not graph_kernels:
        graph_kernels = []
        if has_instance(pipeline_space.values(), CoreGraphGrammar):
            graph_kernels.append("wl")
    graph_kernels = [
        instance_from_map(kernel_map, kernel, "kernel", as_class=True)(
            oa=optimal_assignment,
            se_kernel=instance_from_map(kernel_map, domain_se_kernel, "se kernel"),
        )
        for kernel in graph_kernels
    ]
    hp_kernels = [
        instance_from_map(kernel_map, kernel, "kernel") for kernel in hp_kernels
    ]
    if not graph_kernels and not hp_kernels:
        raise ValueError("No kernels are provided!")
    return graph_kernels, hp_kernels


def instantiate_kernel(pipeline_space, kernels, combine_kernel_type):
    kernel_map = neps.optimizers.bayesian_optimization.kernels.KernelMapping
    combine_kernel_map = (
        neps.optimizers.bayesian_optimization.kernels.CombineKernelMapping
    )

    kernels = [instance_from_map(kernel_map, k, "kernel") for k in (kernels or [])]
    base_kernels = [k for k in kernels if not k.use_as_default]

    combined_kernel = instance_from_map(
        combine_kernel_map, combine_kernel_type, "combine kernel", as_class=True
    )(*base_kernels)

    active_hps = combined_kernel.assign_hyperparameters(dict(pipeline_space))

    # Assign default kernels to HP without ones
    default_kernels = [k for k in kernels if k.use_as_default]
    default_kernels.extend(
        [
            instance_from_map(kernel_map, k, "kernel", kwargs={"active_hps": []})
            for k in DEFAULT_KERNELS
        ]
    )

    for hp_name, hp in pipeline_space.items():
        if not hp_name in active_hps:
            for def_kernel in default_kernels:
                if def_kernel.does_apply_on(hp):
                    def_kernel.active_hps.append(hp_name)
                    break
            else:
                raise ValueError(
                    f"Can't find any default kernel for hyerparameter {hp_name} : {hp}"
                )
    for def_kernel in default_kernels:
        if def_kernel.active_hps:
            combined_kernel.kernels.append(def_kernel)
            combined_kernel.active_hps.extend(def_kernel.active_hps)
    return combined_kernel
