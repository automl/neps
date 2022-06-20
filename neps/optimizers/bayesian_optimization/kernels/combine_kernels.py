from __future__ import annotations

import gpytorch
from metahyper.utils import instance_from_map

import neps

from .base_kernel import Kernel, NoOpKernel


class CombineKernel(Kernel):
    # pylint: disable=abstract-method
    def __init__(self, *kernels, active_hps: None | list = None, **kwargs):
        super().__init__(active_hps=active_hps, **kwargs)
        kernel_map = neps.optimizers.bayesian_optimization.kernels.KernelMapping
        self.kernels = [instance_from_map(kernel_map, k, "kernel") for k in kernels]

    def does_apply_on(self, hp):
        return any([k.does_apply_on(hp) for k in self.kernels])

    def assign_hyperparameters(self, hyperparameters):
        if self.active_hps is not None:
            hyperparameters = {
                hp_name: hp
                for hp_name, hp in hyperparameters.items()
                if hp_name in self.active_hps
            }
        children_hps = set()
        for k in self.kernels:
            children_hps |= k.assign_hyperparameters(hyperparameters)

        self.active_hps = list(children_hps)
        return children_hps

    def _build_sub_kernels(self, hp_shapes):
        sub_kernels = []
        for child_kernel in self.kernels:
            child_shapes = {
                hp_name: hp
                for hp_name, hp in hp_shapes.items()
                if hp_name in child_kernel.active_hps
            }
            child_built_kernel = child_kernel.build(child_shapes)
            if child_built_kernel is not NoOpKernel.NoResult:
                sub_kernels.append(child_built_kernel)
        return sub_kernels


class SumKernel(CombineKernel):
    def build(self, hp_shapes):
        sub_kernels = self._build_sub_kernels(hp_shapes)
        return gpytorch.kernels.AdditiveKernel(*sub_kernels)


class ProductKernel(CombineKernel):
    def build(self, hp_shapes):
        sub_kernels = self._build_sub_kernels(hp_shapes)
        return gpytorch.kernels.ProductKernel(*sub_kernels)
