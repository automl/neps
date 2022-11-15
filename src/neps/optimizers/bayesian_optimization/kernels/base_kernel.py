from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping

import gpytorch
import torch

from ....search_spaces.parameter import HpTensorShape
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
                not assigned to other kernels. It will be used as a fallback kernel by the GP.
            kernel_kwargs: arguments given to the GPyTorch kernel class constructor.
                Can overwrite arguments like the lengthscale_prior value.
            scaling_kernel_kwargs: arguments given to the GPyTorch ScaleKernel constructor.
                Can overwrite arguments like the outputscale_prior value.
        """
        self.active_hps = active_hps
        self.active_hps_shapes: dict[str, HpTensorShape] | HpTensorShape | None = None
        self.use_as_default = use_as_default
        self.kernel_kwargs = kernel_kwargs or {}
        self.scaling_kernel_kwargs = scaling_kernel_kwargs or {}

        if self.use_as_default:
            if self.active_hps is None:
                self.active_hps = []

    @abstractmethod
    def does_apply_on(self, hp):
        """Should return true when the kernel can be used on the given hyperparameter.
        Must be redefined by any child class."""
        raise NotImplementedError

    @abstractmethod
    def build(self, hp_shapes: dict[str, HpTensorShape]):
        """Implementation of the kernel, should return a GPyTorch kernel.
        Must be redefined by any child class.

        Args:
            hp_shapes: for each hyperparemeter of the SearchSapce, an HpTensorShape
                object that can be used to store the size of the tensor representation
                of the HP for this iteration. It might change between iterations."""
        raise Exception()

    def assign_hyperparameters(self, hyperparameters):
        """Assign each HP to one or more kernel. Uses every HP in self.active_hps,
        or every compatible HP if self.active_hps is None."""
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
        """Returns the dimensions of the main configuration tensor this kernel
        will be applied on."""
        active_dims = []
        for shape in hp_shapes.values():
            active_dims.extend(shape.active_dims)
        return tuple(active_dims)

    @staticmethod
    def get_tensor_length(hp_shapes):
        """Returns the number of dimensions of the main configuration tensor this
        kernel will be applied on."""
        return sum(shape.length for shape in hp_shapes.values())

    def _kernel_builder(
        self,
        hp_shapes: dict[str, HpTensorShape],
        kernel_class: gpytorch.kernels.Kernel,
        kernel_kwargs: dict[str, Any] | None = None,
        scaling_kernel_kwargs: dict[str, Any] | None = None,
    ):
        """Build a GPyTorch kernel for one optimizer loop iteration, can be
        called by child instances in the build method. Instanciate a gpytorch kernel
        object with good lengthscale priors and a scale kernel.

        Args:
            hp_shapes: for each hyperparemeter of the SearchSapce, an HpTensorShape
                object that can be used to store the size of the tensor representation
                of the HP for this iteration. It might change between iterations.
            kernel_class: the gpytorch kernel class that will be instanciated.
            kernel_kwargs: to change or add argument for the gpytorch kernel class constructor.
            scaling_kernel_kwargs: to change or add argument for the gpytorch
                ScaleKernel class constructor.
        """
        self.active_hps_shapes = hp_shapes
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


class GenericGPyTorchStationaryKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    is_stationary = True

    def __init__(self, neps_kernel, **kwargs):
        super().__init__(**kwargs)
        self.neps_kernel = neps_kernel

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
        kwargs["training"] = self.training
        return self.neps_kernel.forward(x1, x2, gpytorch_kernel=self, **kwargs)


class CustomKernel(Kernel):
    """Provides a simple way to define a new kernel by overloading the forward
    function only.

    If you inherit from this class, you should only overload the build and
    does_apply_on methods, and you may overload the __init__ method, but not build.
    """

    GenericGPyTorchStationaryKernel = GenericGPyTorchStationaryKernel

    def build(self, hp_shapes: dict[str, HpTensorShape]):
        return self._kernel_builder(
            hp_shapes, self.GenericGPyTorchStationaryKernel, {"neps_kernel": self}
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, gpytorch_kernel, **kwargs
    ) -> torch.Tensor:
        """Should be defined by subclasses as the main kernel function.

        Args:
            x1: tensor representation of one or multiple hyperparameters. The bounds
                between each hyperparameter can be accessed using self.active_hps_shapes.
            x2: tensor representation of the same hyperparameters, with different values.
            kwargs: arguments from GPyTorch
                (see gpytorch documentation for the forward method of a Kernel).
        """
        raise NotImplementedError


class NoOpKernel(Kernel):
    class NoResult:
        pass

    def __init__(self, *args, restrict_to: list = None, **kwargs):
        """Build a Kernel that does nothing. If 'restrict_to' is not None,
        it should be a list of types this kernel will be applied on. Usefull
        for default kernels.
        """
        super().__init__(*args, **kwargs)
        self.restrict_to = restrict_to

    def does_apply_on(self, hp):
        if self.restrict_to is None:
            return True
        for kernel_cls in self.restrict_to:
            if isinstance(hp, kernel_cls):
                return True
        return False

    def build(self, hp_shapes):  # pylint: disable=unused-argument
        return self.NoResult
