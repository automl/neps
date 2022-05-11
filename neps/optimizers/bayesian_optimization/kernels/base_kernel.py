from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Callable

import botorch
import gpytorch
import torch
from metahyper.utils import instance_from_map

from ....search_spaces import CategoricalParameter, NumericalParameter

# Static kernels


class Kernel:
    def __init__(
        self,
        active_hps: None | list = None,
        kernel_kwargs=None,
    ):
        self.active_hps = active_hps
        self.kernel_kwargs = kernel_kwargs or {}

    @abstractmethod
    def does_apply_on(self, hp):
        raise NotImplementedError

    @abstractmethod
    def build(self, hp_shapes):
        raise Exception()

    def assign_hyperparameters(self, hyperparameters):
        if self.active_hps is None:
            self.active_hps = [
                hp_name
                for hp_name, hp in hyperparameters.items()
                if self.does_apply_on(hp)
            ]
        if not self.active_hps:
            raise Exception("Can't build a kernel without hyperparameters to apply on")
        return set(self.active_hps)

    @staticmethod
    def get_active_dims(hp_shapes):
        active_dims = []
        for shape in hp_shapes.values():
            active_dims.extend(shape.active_dims)
        return tuple(active_dims)

    @staticmethod
    def get_tensor_length(hp_shapes):
        return sum(shape.length for shape in hp_shapes.values())


class StationaryKernel(Kernel):
    # pylint: disable=abstract-method
    def __init__(
        self,
        ard_num_dims: None | float = None,
        scale_kwargs=None,
        **kwargs,
    ):
        self.ard_num_dims = ard_num_dims
        self.scale_kwargs = scale_kwargs or {}
        super().__init__(**kwargs)

    def _kernel_builder(
        self, hp_shapes, kernel_class, kernel_kwargs=None, scale_kwargs=None
    ):
        return gpytorch.kernels.ScaleKernel(
            kernel_class(
                **{
                    "ard_num_dims": self.get_tensor_length(hp_shapes),
                    "active_dims": self.get_active_dims(hp_shapes),
                    "lengthscale_prior": gpytorch.priors.GammaPrior(3.0, 6.0),
                    **(kernel_kwargs or {}),
                    **self.kernel_kwargs,
                }
            ),
            **{
                "outputscale_prior": gpytorch.priors.GammaPrior(2.0, 0.15),
                **(scale_kwargs or {}),
                **self.scale_kwargs,
            },
        )


class BaseNumericalKernel(StationaryKernel):
    # pylint: disable=abstract-method
    def does_apply_on(self, hp):
        return isinstance(hp, NumericalParameter) and not isinstance(
            hp, CategoricalParameter
        )


class MaternKernel(BaseNumericalKernel):
    def __init__(self, nu: float = 2.5, **kwargs):
        self.nu = nu
        super().__init__(**kwargs)

    def build(self, hp_shapes):
        return self._kernel_builder(
            hp_shapes, gpytorch.kernels.MaternKernel, {"nu": self.nu}
        )


class BaseCategoricalKernel(StationaryKernel):
    # pylint: disable=abstract-method
    def does_apply_on(self, hp):
        return isinstance(hp, CategoricalParameter)


class CategoricalBotorchKernel(BaseCategoricalKernel):
    def build(self, hp_shapes):
        return self._kernel_builder(
            hp_shapes,
            botorch.models.kernels.CategoricalKernel,
        )


class GptHammingKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    is_stationary = True

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
        assert not kwargs["diag"] and not kwargs["last_dim_is_batch"]

        indicator = x1.unsqueeze(1) != x2
        K = (-1 / (2 * self.lengthscale**2) * indicator).sum(axis=2)
        return torch.exp(K)


class HammingKernel(BaseCategoricalKernel):
    def build(self, hp_shapes):
        return self._kernel_builder(hp_shapes, GptHammingKernel)


# Combine kernels


class CombineKernel(Kernel):
    # pylint: disable=abstract-method
    def __init__(self, *kernels, active_hps: None | list = None, **kwargs):
        super().__init__(active_hps=active_hps, **kwargs)
        self.kernels = [instance_from_map(KernelMapping, k, "kernel") for k in kernels]

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
            sub_kernels.append(child_kernel.build(child_shapes))
        return sub_kernels


class SumKernel(CombineKernel):
    def build(self, hp_shapes):
        sub_kernels = self._build_sub_kernels(hp_shapes)
        return gpytorch.kernels.AdditiveKernel(*sub_kernels)


class ProductKernel(CombineKernel):
    def build(self, hp_shapes):
        sub_kernels = self._build_sub_kernels(hp_shapes)
        return gpytorch.kernels.ProductKernel(*sub_kernels)


# TODO : move to __init__.py
CombineKernelMapping: dict[str, Callable] = {
    "sum": SumKernel,
    "product": ProductKernel,
}

KernelMapping: dict[str, Callable] = {
    "m32": partial(MaternKernel, nu=3 / 2),
    "m52": partial(MaternKernel, nu=5 / 2),
    "categorical": CategoricalBotorchKernel,
    "hm": HammingKernel,
    **CombineKernelMapping,
}
