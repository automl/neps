from __future__ import annotations

import gpytorch
import torch

import neps
from metahyper.utils import instance_from_map

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

    def _build_sub_kernels(self, hp_shapes: dict) -> list:
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

    def assign_hierarichal_hyperparameters(self, hirarichal_hps_and_levels: list):
        """
        A trick to isolate kernels for each hierarchical level, as opposed to
        all graph kernels being used for all graph data.
        """
        h_l_idx = 0
        for k in self.kernels:
            if h_l_idx < len(hirarichal_hps_and_levels):
                hp_name = hirarichal_hps_and_levels[h_l_idx]
                if hp_name in k.active_hps:
                    k.active_hps = [hp_name]
                    h_l_idx += 1
            else:
                for hp_n in hirarichal_hps_and_levels:
                    if hp_n in k.active_hps:
                        k.active_hps.remove(hp_n)
        self.assign_hyperparameters({})

    def assign_feature_hyperparameters(self, feature_names: list):
        """
        A trick to isolate the kernel for the graph feature data, as opposed to
        using different Float kernels for graph feature data
        """
        h_idx = 0
        for k in self.kernels:
            if h_idx < len(feature_names):
                hp_name = feature_names[h_idx]
                if hp_name in k.active_hps:
                    k.active_hps = [hp_name]
                    h_idx += 1
            else:
                for hp_n in feature_names:
                    if hp_n in k.active_hps:
                        k.active_hps.remove(hp_n)
        self.assign_hyperparameters({})


class AdditiveKernel(gpytorch.kernels.AdditiveKernel):
    """
    Extends gpytorch.kernels.AdditiveKernel to include scalar weights
    for each kernel, which will be optimized as well

    Each weight is multiplied by a single kernel output.
    """

    def __init__(self, *kernels):
        super().__init__(*kernels)
        self.kernel_weights = torch.tensor([1.0 / len(self.kernels)] * len(self.kernels))

    def forward(self, x1, x2, diag=False, **params):
        res = gpytorch.lazy.ZeroLazyTensor() if not diag else 0
        for idx, kern in enumerate(self.kernels):
            next_term = kern(x1, x2, diag=diag, **params) * self.kernel_weights[idx]
            if not diag:
                res = res + gpytorch.lazy.lazify(next_term)
            else:
                res = res + next_term

        return res


class SumKernel(CombineKernel):
    def build(self, hp_shapes):
        sub_kernels = self._build_sub_kernels(hp_shapes)
        return AdditiveKernel(*sub_kernels)


class ProductKernel(CombineKernel):
    def build(self, hp_shapes):
        sub_kernels = self._build_sub_kernels(hp_shapes)
        return gpytorch.kernels.ProductKernel(*sub_kernels)
