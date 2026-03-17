from __future__ import annotations

import math
import pytest
import torch
from pytest_cases import parametrize

from neps.space import Domain

T = torch.tensor


@parametrize(
    "x, frm, expected",
    [
        # Remains unchanged if from unit-float
        (T([0, 0.5, 1.0]), Domain.unit_float(), T([0, 0.5, 1.0])),
        # Converts integers to float
        (T([0, 1]), Domain.unit_float(), T([0.0, 1.0])),
        # Integer conversion
        (T([0, 1, 2, 3, 4]), Domain.integer(0, 4), T([0.0, 0.25, 0.5, 0.75, 1.0])),
        # Negatives
        (
            T([-0.5, -0.25, 0.0, 0.25, 0.5]),
            Domain.floating(-0.5, 0.5),
            T([0.0, 0.25, 0.5, 0.75, 1.0]),
        ),
        # Log scale
        (
            T([1e-4, 1e-3, 1e-2, 1e-1, 1]),
            Domain.floating(1e-4, 1, log=True),
            T([0.0, 0.25, 0.5, 0.75, 1.0]),
        ),
        # Binned
        (
            torch.arange(10),
            Domain.integer(0, 10, bins=5),
            T([0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0]),
        ),
    ],
)
def test_domain_to_unit(x: torch.Tensor, frm: Domain, expected: torch.Tensor) -> None:
    y = frm.to_unit(x)
    assert y.dtype == torch.float64
    torch.testing.assert_close(y, expected, check_dtype=False, msg=f"{y} != {expected}")


def test_domain_to_unit_dtype_with_floating() -> None:
    domain = Domain.integer(0, 4)
    x = T([0, 1, 2, 3, 4], dtype=torch.int32)

    expected_64 = T([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    y_64 = domain.to_unit(x, dtype=torch.float64)
    torch.testing.assert_close(y_64, expected_64, check_dtype=True)

    expected_32 = T([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    y_32 = domain.to_unit(x, dtype=torch.float32)
    torch.testing.assert_close(y_32, expected_32, check_dtype=True)


def test_domain_to_unit_dtype_with_integer_fails() -> None:
    domain = Domain.integer(0, 4)
    x = T([0, 1, 2, 3, 4], dtype=torch.int32)

    with pytest.raises(ValueError, match="only allows floating dtypes"):
        domain.to_unit(x, dtype=torch.int32)


@parametrize(
    "x, to, expected",
    [
        # Remains unchanged if from unit-float
        (
            T([0, 0.5, 1.0]),
            Domain.unit_float(),
            T([0, 0.5, 1.0], dtype=torch.float64),
        ),
        # Converts floats to integers
        (
            T([0.0, 1.0]),
            Domain.integer(0, 1),
            T([0, 1], dtype=torch.int64),
        ),
        # Integer range
        (
            T([0, 0.25, 0.5, 0.75, 1.0]),
            Domain.integer(0, 4),
            T([0, 1, 2, 3, 4], dtype=torch.int64),
        ),
        # Negatives
        (
            T([0.0, 0.25, 0.5, 0.75, 1.0]),
            Domain.floating(-0.5, 0.5),
            T([-0.5, -0.25, 0.0, 0.25, 0.5], dtype=torch.float64),
        ),
        # Log scale
        (
            T([0.0, 0.25, 0.5, 0.75, 1.0]),
            Domain.floating(1e-4, 1, log=True),
            T([1e-4, 1e-3, 1e-2, 1e-1, 1], dtype=torch.float64),
        ),
        # Binned
        (
            T([0.0, 0.25, 0.5, 0.75, 1.0]),
            Domain.integer(0, 20, bins=5),
            T([0, 5, 10, 15, 20], dtype=torch.int64),
        ),
    ],
)
def test_domain_from_unit(x: torch.Tensor, to: Domain, expected: torch.Tensor) -> None:
    x = x.to(dtype=torch.float64)
    y = to.from_unit(x)
    torch.testing.assert_close(y, expected, check_dtype=True, msg=f"{y} != {expected}")


def test_domain_from_unit_dtype() -> None:
    x = T([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    domain = Domain.integer(0, 4)

    expected_f64 = T([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    y_f64 = domain.from_unit(x, dtype=torch.float64)
    torch.testing.assert_close(y_f64, expected_f64, check_dtype=True)

    expected_f32 = T([0, 1, 2, 3, 4], dtype=torch.float32)
    y_f32 = domain.from_unit(x, dtype=torch.float32)
    torch.testing.assert_close(y_f32, expected_f32, check_dtype=True)

    expected_i32 = T([0, 1, 2, 3, 4], dtype=torch.int32)
    y_i32 = domain.from_unit(x, dtype=torch.int32)
    torch.testing.assert_close(y_i32, expected_i32, check_dtype=True)

    expected_i64 = T([0, 1, 2, 3, 4], dtype=torch.int64)
    y_i64 = domain.from_unit(x, dtype=torch.int64)
    torch.testing.assert_close(y_i64, expected_i64, check_dtype=True)


@parametrize(
    "x, frm, to, expected",
    [
        (
            T([1e-2, 1e-1, 1e0, 1e1, 1e2], dtype=torch.float64),
            Domain.floating(1e-2, 1e2, log=True),
            Domain.floating(-2, 2),
            T([-2, -1, 0, 1, 2], dtype=torch.float64),
        ),
        (
            T([0, 2, 4, 6, 8], dtype=torch.int64),
            Domain.integer(0, 8, bins=5),
            Domain.integer(0, 4),
            T([0, 1, 2, 3, 4], dtype=torch.int64),
        ),
        (
            T([10, 12.5, 15], dtype=torch.float64),
            Domain.floating(10, 15),
            Domain.floating(2, 3),
            T([2, 2.5, 3.0], dtype=torch.float64),
        ),
    ],
)
def test_domain_casting(
    x: torch.Tensor, frm: Domain, to: Domain, expected: torch.Tensor
) -> None:
    y = to.cast(x, frm=frm)
    torch.testing.assert_close(y, expected, check_dtype=True, msg=f"{y} != {expected}")

    x_back = frm.cast(y, frm=to)
    torch.testing.assert_close(x_back, x, check_dtype=True, msg=f"{x_back} != {x}")


@parametrize(
    "x, frm, to, expected",
    [
        (
            # Combines all the previous cast domains in one go as a single tensor
            T(
                [
                    [1e-2, 1e-1, 1e0, 1e1, 1e2],
                    [0, 2, 4, 6, 8],
                    [10, 12.5, 15, 17.5, 20],
                ]
            ).transpose(0, 1),
            [
                Domain.floating(1e-2, 1e2, log=True),
                Domain.integer(0, 8, bins=5),
                Domain.floating(10, 20),
            ],  # from
            [Domain.floating(-2, 2), Domain.integer(0, 4), Domain.floating(2, 4)],  # to
            T(
                [
                    [-2, -1, 0, 1, 2],
                    [0, 1, 2, 3, 4],
                    [2, 2.5, 3, 3.5, 4],
                ]
            ).transpose(0, 1),
        ),
        (
            # This was a random case found while testing samplers which seemed to fail
            # Uniform noise convert to integers
            # 0-0.25 -> 12,
            # 0.25-0.5 -> 13,
            # 0.5-0.75 -> 14
            # 0.75-1 -> 15
            T(
                [
                    [0.2350, 0.6488, 0.6411],
                    [0.6457, 0.2897, 0.6879],
                    [0.7401, 0.4268, 0.7607],
                ]
            ),
            Domain.unit_float(),
            Domain.integer(12, 15),
            T(
                [
                    [12, 14, 14],
                    [14, 13, 14],
                    [14, 13, 15],
                ]
            ),
        ),
    ],
)
def test_translate(
    x: torch.Tensor,
    frm: list[Domain],
    to: list[Domain],
    expected: torch.Tensor,
) -> None:
    y = Domain.translate(x, frm=frm, to=to)
    torch.testing.assert_close(y, expected, check_dtype=True, msg=f"{y} != {expected}")


def test_log_base_2_to_unit() -> None:
    """Test converting values with log base 2 to unit interval."""
    # Domain: 2^0 to 2^8 (1 to 256)
    # Values: 1, 2, 4, 8, 16, 32, 64, 128, 256
    # log2 values: 0, 1, 2, 3, 4, 5, 6, 7, 8
    # normalized to [0, 1]: 0/8, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8
    domain = Domain.integer(1, 256, log=True, log_base=2)
    x = T([1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=torch.float64)
    expected = T([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    
    y = domain.to_unit(x)
    torch.testing.assert_close(y, expected, check_dtype=False, atol=1e-6, rtol=1e-6)


def test_log_base_10_to_unit() -> None:
    """Test converting values with log base 10 to unit interval."""
    # Domain: 10^-2 to 10^2 (0.01 to 100)
    # Values: 0.01, 0.1, 1, 10, 100
    # log10 values: -2, -1, 0, 1, 2
    # normalized to [0, 1]: 0/4, 1/4, 2/4, 3/4, 4/4
    domain = Domain.floating(0.01, 100, log=True, log_base=10)
    x = T([0.01, 0.1, 1.0, 10.0, 100.0])
    expected = T([0.0, 0.25, 0.5, 0.75, 1.0])
    
    y = domain.to_unit(x)
    torch.testing.assert_close(y, expected, check_dtype=False, atol=1e-6, rtol=1e-6)


def test_log_base_e_natural_log() -> None:
    """Test that log_base=e gives same result as no log_base (natural log)."""
    import math
    
    domain_natural = Domain.floating(1.0, math.e**4, log=True, log_base=None)
    
    x = T([1.0, math.e, math.e**2, math.e**3, math.e**4])
    x_expected = T([0.0, 1/4, 2/4, 3/4, 1.0])
    
    y_natural = domain_natural.to_unit(x)
    
    torch.testing.assert_close(y_natural, x_expected, check_dtype=False, atol=1e-6, rtol=1e-6)


def test_log_base_2_from_unit() -> None:
    """Test converting from unit interval back to log base 2 domain."""
    domain = Domain.integer(1, 256, log=True, log_base=2)
    x = T([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], dtype=torch.float64)
    expected = T([1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=torch.int64)
    
    y = domain.from_unit(x)
    torch.testing.assert_close(y, expected, check_dtype=True, atol=1e-5, rtol=1e-5)


def test_log_base_10_from_unit() -> None:
    """Test converting from unit interval back to log base 10 domain."""
    domain = Domain.floating(0.01, 100, log=True, log_base=10)
    x = T([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    expected = T([0.01, 0.1, 1.0, 10.0, 100.0])
    
    y = domain.from_unit(x)
    torch.testing.assert_close(y, expected, check_dtype=False, atol=1e-6, rtol=1e-6)


def test_log_base_round_trip() -> None:
    """Test that to_unit and from_unit are inverses with log_base."""
    import math
    
    domain = Domain.floating(10, 10000, log=True, log_base=10)
    original = T([10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0])
    
    # to_unit and back
    normalized = domain.to_unit(original)
    reconstructed = domain.from_unit(normalized)
    
    torch.testing.assert_close(reconstructed, original, check_dtype=False, atol=1e-5, rtol=1e-5)


def test_log_base_2_vs_natural_different() -> None:
    """Verify that log_bounds differ between different log bases."""
    domain_base2 = Domain.floating(1.0, 256.0, log=True, log_base=2)
    domain_natural = Domain.floating(1.0, 256.0, log=True, log_base=None)
    
    # Log bounds should be different
    assert domain_base2.log_bounds != domain_natural.log_bounds
    
    # Base 2: log_bounds should be (0, 8) since log2(1)=0, log2(256)=8
    assert abs(domain_base2.log_bounds[0] - 0.0) < 1e-10
    assert abs(domain_base2.log_bounds[1] - 8.0) < 1e-10
    
    # Natural log: log_bounds should be (0, ~5.545) since ln(1)=0, ln(256)≈5.545
    assert abs(domain_natural.log_bounds[0] - 0.0) < 1e-10
    assert abs(domain_natural.log_bounds[1] - math.log(256)) < 1e-10


def test_log_base_casting_with_different_bases() -> None:
    """Test casting between domains with different log bases."""
    # From log base 2, to log base 10
    domain_from = Domain.floating(1.0, 1024.0, log=True, log_base=2)
    domain_to = Domain.floating(1.0, 1024.0, log=True, log_base=10)
    
    x = T([1.0, 32.0, 1024.0])
    
    # Cast through unit interval
    y = domain_to.cast(x, frm=domain_from)
    
    # All values should remain the same (same original domain just different base)
    torch.testing.assert_close(y, x, check_dtype=False, atol=1e-5, rtol=1e-5)
