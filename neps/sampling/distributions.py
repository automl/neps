"""Custom distributions for NEPS."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING, ClassVar, Mapping
from typing_extensions import override

import torch
from torch.distributions import Distribution, Uniform, constraints
from torch.distributions.utils import broadcast_all

from neps.search_spaces.domain import Domain

if TYPE_CHECKING:
    from neps.search_spaces.architecture.cfg_variants.constrained_cfg import Constraint

CONST_SQRT_2 = torch.tensor(math.sqrt(2), dtype=torch.float64)
CONST_INV_SQRT_2PI = torch.tensor(1 / math.sqrt(2 * math.pi), dtype=torch.float64)
CONST_INV_SQRT_2 = torch.tensor(1 / math.sqrt(2), dtype=torch.float64)
CONST_LOG_INV_SQRT_2PI = torch.tensor(math.log(CONST_INV_SQRT_2PI), dtype=torch.float64)
CONST_LOG_SQRT_2PI_E = torch.tensor(
    0.5 * math.log(2 * math.pi * math.e),
    dtype=torch.float64,
)

# from https://github.com/toshas/torch_truncnorm


class TruncatedStandardNormal(Distribution):
    """Truncated Standard Normal distribution.

    Source: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints: ClassVar[Mapping[str, Constraint]] = {
        "a": constraints.real,
        "b": constraints.real,
    }  # type: ignore
    has_rsample: ClassVar[bool] = True
    eps: ClassVar[float] = 1e-6

    def __init__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        validate_args: bool | None = None,
        device: torch.device | None = None,
    ):
        """Initialize a truncated standard normal distribution.

        Args:
            a: Lower truncation bound.
            b: Upper truncation bound.
            validate_args: Whether to validate input.
            device: Device to use.
        """
        self.a, self.b = broadcast_all(a, b)
        self.a = self.a.to(device)
        self.b = self.b.to(device)

        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()

        super().__init__(batch_shape, validate_args=validate_args)

        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")

        if any((self.a >= self.b).view(-1).tolist()):
            raise ValueError("Incorrect truncation range")

        eps = self.eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp(eps, 1 - eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b
            - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    @override
    def support(self) -> constraints._Interval:
        return constraints.interval(self.a, self.b)

    @property
    @override
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    @override
    def variance(self) -> torch.Tensor:
        return self._variance

    @override
    def entropy(self) -> torch.Tensor:
        return self._entropy

    @staticmethod
    def _little_phi(x: torch.Tensor) -> torch.Tensor:
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    def _big_phi(self, x: torch.Tensor) -> torch.Tensor:
        phi = 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())
        return phi.clamp(self.eps, 1 - self.eps)

    @staticmethod
    def _inv_big_phi(x: torch.Tensor) -> torch.Tensor:
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    @override
    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    @override
    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        y = self._big_phi_a + value * self._Z
        y = y.clamp(self.eps, 1 - self.eps)
        return self._inv_big_phi(y)

    @override
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    @override
    def rsample(self, sample_shape: torch.Size | None = None) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """Truncated Normal distribution.

    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    def __init__(
        self,
        loc: float | torch.Tensor,
        scale: float | torch.Tensor,
        a: float | torch.Tensor,
        b: float | torch.Tensor,
        validate_args: bool | None = None,
        device: torch.device | None = None,
    ):
        """Initialize a truncated standard normal distribution.

        Args:
            loc: The mean of the distribution.
            scale: The std of the distribution.
            a: The lower bound of the distribution.
            b: The upper bound of the distribution.
            validate_args: Whether to validate input.
            device: Device to use.
        """
        scale = torch.as_tensor(scale, device=device)
        scale = scale.clamp_min(self.eps)

        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = a.to(device)  # type: ignore
        b = b.to(device)  # type: ignore
        self._non_std_a = a
        self._non_std_b = b
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super().__init__(a, b, validate_args=validate_args)  # type: ignore
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    @override
    def cdf(self, value):
        return super().cdf(self._to_std_rv(value))

    @override
    def icdf(self, value):
        sample = self._from_std_rv(super().icdf(value))

        # clamp data but keep gradients
        sample_clip = torch.stack(
            [sample.detach(), self._non_std_a.detach().expand_as(sample)], 0
        ).max(0)[0]
        sample_clip = torch.stack(
            [sample_clip, self._non_std_b.detach().expand_as(sample)], 0
        ).min(0)[0]
        sample.data.copy_(sample_clip)
        return sample

    @override
    def log_prob(self, value):
        value = self._to_std_rv(value)
        return super().log_prob(value) - self._log_scale


class UniformWithUpperBound(Uniform):
    """Uniform distribution with upper bound inclusive.

    This is mostly a hack because torch's version of Uniform does not include
    the upper bound which only causes a problem when considering the log_prob.
    Otherwise the upper bound works with every other method.
    """

    # OPTIM: This could probably be optimized a lot but I'm not sure how it effects
    # gradients. Could probably do a different path depending on if `value` requires
    # gradients or not.
    @override
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

        lb = self.low.le(value).type_as(self.low)
        ub = self.high.ge(value).type_as(self.low)  # The main change, is `gt` in original
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)


@dataclass
class TorchDistributionWithDomain:
    distribution: Distribution
    domain: Domain


UNIT_UNIFORM_DIST = TorchDistributionWithDomain(
    distribution=UniformWithUpperBound(0, 1),
    domain=Domain.unit_float(),
)

if __name__ == "__main__":
    loc = 0.95
    for confidence in torch.linspace(0.0, 0.8, 8):
        scale = 1 - confidence
        dist = TruncatedNormal(
            loc=loc,
            scale=scale,
            a=0.0,
            b=1.0,
        )
        xs = torch.linspace(0, 1, 100)
        ys = dist.log_prob(xs)
        import matplotlib.pyplot as plt

        plt.plot(xs, ys, label=f"confidence={confidence}")
        plt.plot(loc, dist.log_prob(torch.tensor(loc)), "ro")
    plt.legend()
    plt.show()
