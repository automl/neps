from __future__ import annotations

import math
import inspect
import copy
from typing import TypeVar, Generic, Any, Sequence, Mapping, Callable
from typing_extensions import Self
import torch
import torch.nn as nn

from neps.utils.types import NotSet

T = TypeVar("T")


class Kernel(nn.Module, Generic[T]):
    def fit_transform(self, x: T) -> torch.Tensor:
        raise NotImplementedError

    def transform(self, x: T) -> torch.Tensor:
        raise NotImplementedError

    def clone(self) -> Self:
        return self.clone_with()

    def clone_with(self, **params: dict[str, Any]) -> Self:
        # h ttps://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/base.py#L197
        sig = inspect.signature(self.__init__)

        self_values = {}
        for p in sig.parameters.values():
            if p.name == "self":
                continue

            attr = getattr(self, p.name, NotSet)
            if attr is NotSet:
                raise ValueError(
                    f"Could not clone as the variable {p.name} was not set in"
                    f" the constructor on the object: {self}"
                )
            self_values[p.name] = params.get(p.name, attr)

        new_self_values = copy.deepcopy(self_values)
        return self.__class__(**new_self_values)

    def grid_search(
        self,
        x: T,
        *,
        grid: Sequence[Mapping[str, Any]],
        to_minimize: Callable[[torch.Tensor], float],
    ) -> tuple[Self, float]:
        if len(grid) == 0:
            raise ValueError("Grid must have at least one element.")

        def _fit_and_eval(_params: Mapping[str, Any]) -> tuple[Kernel[T], float]:
            cloned_kernel = self.clone_with(**_params)
            K = cloned_kernel.fit_transform(x)
            metric = to_minimize(K)
            return cloned_kernel, metric

        return min(
            (_fit_and_eval(params) for params in grid),
            key=lambda x: x[1],
        )


class NumericKernel(Kernel[torch.Tensor]): ...


PI = torch.tensor(math.pi)


def compute_normalized_log_marginal_likelihood(
    K_i: torch.Tensor,
    logDetK: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute the zero mean Gaussian process log marginal likelihood
    given the inverse of Gram matrix K(x2,x2), its log determinant,
    and the training label vector y.
    """
    lml = -0.5 * (y.t() @ K_i @ y) + 0.5 * logDetK - y.shape[0] / 2.0 * torch.log(2 * PI)
    return lml / y.shape[0]


def compute_pd_inverse(
    K: torch.Tensor,
    *,
    jitter: float | torch.Tensor = 1e-9,
    attempts: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the inverse of a postive-(semi)definite matrix K using Cholesky inversion."""
    n = K.shape[0]
    assert (
        isinstance(jitter, float) or jitter.ndim == 0
    ), "only homoscedastic noise variance is allowed here!"
    for i in range(attempts):
        try:
            jitter_diag = jitter * torch.eye(n, device=K.device) * 10**i
            Kc = torch.linalg.cholesky(K + jitter_diag)
            break
        except RuntimeError:
            pass
    else:
        raise RuntimeError(f"Gram matrix not positive definite despite of jitter:\n{K}")

    logDetK = -2 * torch.sum(torch.log(torch.diag(Kc)))
    K_i = torch.cholesky_inverse(Kc)
    return K_i.to(dtype=torch.float64), logDetK.to(dtype=torch.float64)
