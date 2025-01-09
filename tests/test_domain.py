from __future__ import annotations

import pytest
import torch
from pytest_cases import parametrize

from neps.search_spaces.domain import Domain

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
            # This test combines all the previous cast domains in one go as a single tensor
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
