from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
import math
from typing import Any, ClassVar, Generic, Mapping, Sequence, TypeVar
from typing_extensions import Self

import torch
from torch import nn

from neps.utils.types import NotSet

T = TypeVar("T")


class Kernel(ABC, nn.Module, Generic[T]):
    suggested_grid: ClassVar[Sequence[Mapping[str, Any]]]

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def as_optimizable(self) -> Self: ...

    @abstractmethod
    def forward(self, x: T, x2: T | None = None) -> torch.Tensor:
        raise NotImplementedError

    def clone(self) -> Self:
        return self.clone_with()

    def clone_with(self, **params: Any) -> Self:
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
        y: torch.Tensor,
        *,
        grid: Sequence[Mapping[str, Any]],
        noise_variances: Sequence[float] = (1e-6,),
    ) -> tuple[Self, float] | Exception:
        # Returns: (Kernel[T], float) | None if failed
        if len(grid) == 0:
            raise ValueError("Grid must have at least one element.")

        def _fit_and_eval(
            _params: Mapping[str, Any],
        ) -> tuple[Kernel[T], float] | Exception:
            cloned_kernel = self.clone_with(**_params)
            K = cloned_kernel.forward(x)

            best_lml = -float("inf")
            exception: Exception | None = None
            for noise_variance in noise_variances:
                K.diag().add_(noise_variance)

                K_inv, logDetK = compute_pd_inverse(K)
                lml = log_marginal_likelihood(K_inv, logDetK, y).item()
                if lml > best_lml:
                    best_lml = lml

                K.diag().sub_(noise_variance)

            if exception is None:
                return cloned_kernel, best_lml

            return exception

        evals = [_fit_and_eval(params) for params in grid]
        evals_with_score = [e for e in evals if not isinstance(e, Exception)]
        if not any(evals_with_score):
            raise evals[-1]  # type: ignore

        best_eval = max(evals_with_score, key=lambda e: e[1])  # type: ignore
        return best_eval


class NumericKernel(Kernel[torch.Tensor]): ...


TWO_LOG_2_PI = 2 * torch.log(torch.tensor(2 * math.pi))


def log_marginal_likelihood(
    K_inv: torch.Tensor,
    logDetK: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # y.T @ K_inv @ y  --- Benchmarked to be twice as fast
    quad_form = torch.matmul(y, torch.matmul(K_inv, y))
    n = y.shape[0]

    # TODO: We can drop the `n / 2 * TWO_LOG_2_PI` term for the grid
    # search above as it's constant between the different kernel grids
    # as it's purely data dependant with the `n`
    return -0.5 * quad_form + 0.5 * logDetK - n / TWO_LOG_2_PI


class _CholeskyError(RuntimeError):
    """Raised when the Cholesky decomposition fails."""


# https://github.com/cornellius-gp/linear_operator/blob/eec70f9e1cd9106c32b05a3e774ea29d00d71cea/linear_operator/utils/cholesky.py#L12
def _cholesky_routine(
    K: torch.Tensor,
    jitter: float | torch.Tensor = 1e-6,
    max_tries: int = 4,
) -> torch.Tensor:
    L, info = torch.linalg.cholesky_ex(K)
    if not torch.any(info):
        return L

    # Clone as we will modify in place, still cheaper
    # than creating a new full tensor for identity.
    K_prime = K.clone()
    jitter_prev = 0
    for i in range(max_tries):
        jitter_new = jitter * (10**i)
        K_prime.diagonal().add_(jitter_new - jitter_prev)
        L, info = torch.linalg.cholesky_ex(K_prime)
        if not torch.any(info):
            return L

        jitter_prev = jitter_new

    raise _CholeskyError("Failed to compute Cholesky decomposition.")


def compute_pd_inverse(K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Adding noise to the diagonal of K helps with numerical stability
    # when K is singular or near-singular, (i.e. it helps K be more "positive") which
    # is required for the decomposition.

    try:
        # L @ L.T = K_inv  --- solves for L
        L = _cholesky_routine(K)
        logDetK = 2 * torch.sum(torch.log(torch.diag(L)))

        # K_inv = L_inv @ L_inv.T  --- Efficiently solve for K_inv using just L
        K_inv = torch.cholesky_inverse(L)
    except _CholeskyError:
        # If we fail to compute the Cholesky decomposition,
        # then just compute the inverse directly.
        K_inv = torch.linalg.inv(K)
        logDetK = torch.linalg.slogdet(K)[1]

    return K_inv, logDetK
