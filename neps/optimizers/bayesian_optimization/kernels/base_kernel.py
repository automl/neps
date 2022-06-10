from __future__ import annotations

from abc import abstractmethod
from typing import Mapping

import gpytorch
import torch

from ..default_consts import LENGTHSCALE_MAX, LENGTHSCALE_MIN
from ..utils import SafeInterval


class Kernel:
    has_scaling = True

    def __init__(
        self,
        active_hps: None | list = None,
        use_as_default: bool = False,
        kernel_kwargs: Mapping | None = None,
        scaling_kernel_kwargs: Mapping | None = None,
    ):
        """Base Kernel class

        Args:
            active_hps: name of the hyperparameters on which the kernel will be applied.
                If not specified, it will be applied on all hyperparameters with a matching type.
            use_as_default: if True, the Kernel will be applied only to hyperparameters
                not assigned to other kernels
            kernel_kwargs: arguments given to the GPyTorch kernel class constructor.
                Can overwrite arguments like the lengthscale_prior value.
            scaling_kernel_kwargs: arguments given to the GPyTorch ScaleKernel constructor.
                Can overwrite arguments like the outputscale_prior value.
        """
        self.active_hps = active_hps
        self.use_as_default = use_as_default
        self.kernel_kwargs = kernel_kwargs or {}
        self.scaling_kernel_kwargs = scaling_kernel_kwargs or {}

        if self.use_as_default:
            if self.active_hps is None:
                self.active_hps = []

    @abstractmethod
    def does_apply_on(self, hp):
        """Should return true when the kernel can be used on the given hyperparameter"""
        raise NotImplementedError

    @abstractmethod
    def build(self, hp_shapes):
        """Implementation of the kernel, should return a GPyTorch kernel."""
        raise Exception()

    def assign_hyperparameters(self, hyperparameters):
        if self.active_hps is None:
            self.active_hps = [
                hp_name
                for hp_name, hp in hyperparameters.items()
                if self.does_apply_on(hp)
            ]
        if not self.active_hps:
            raise Exception(
                f"Can't build the {self.__class__.__name__} kernel without hyperparameters to apply on"
            )
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

    def _kernel_builder(
        self, hp_shapes, kernel_class, kernel_kwargs=None, scaling_kernel_kwargs=None
    ):
        kernel = kernel_class(
            **{
                "ard_num_dims": self.get_tensor_length(hp_shapes),
                "active_dims": self.get_active_dims(hp_shapes),
                "lengthscale_prior": gpytorch.priors.NormalPrior(0.09, 0.1),
                "lengthscale_constraint": SafeInterval(
                    LENGTHSCALE_MIN,
                    LENGTHSCALE_MAX,
                ),
                **(kernel_kwargs or {}),
                **self.kernel_kwargs,
            }
        )
        if self.has_scaling:
            kernel = gpytorch.kernels.ScaleKernel(
                kernel,
                **{
                    "outputscale_prior": gpytorch.priors.GammaPrior(2.0, 0.15),
                    **(scaling_kernel_kwargs or {}),
                    **self.scaling_kernel_kwargs,
                },
            )
        return kernel


class CustomKernel(Kernel):
    """Provides a simple way to define a new kernel by overloading the forward function only"""

    class GenericGPyTorchStationaryKernel(gpytorch.kernels.Kernel):
        has_lengthscale = True
        is_stationary = True

        def __init__(self, neps_kernel, **kwargs):
            super().__init__(**kwargs)
            self.neps_kernel = neps_kernel

        def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
            return self.neps_kernel.forward(x1, x2, gpytorch_kernel=self, **kwargs)

    def build(self, hp_shapes):
        return self._kernel_builder(
            hp_shapes, self.GenericGPyTorchStationaryKernel, {"neps_kernel": self}
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, gpytorch_kernel, **kwargs
    ) -> torch.Tensor:
        """Should be defined by subclasses"""
        raise NotImplementedError
