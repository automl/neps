from __future__ import annotations

from typing import Any

import pytest

from neps.space import Categorical, Constant, Float, Integer, Parameter, parsing


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            (0, 1),
            Integer(0, 1),
        ),
        (
            ("1e3", "1e5"),
            Integer(1e3, 1e5),
        ),
        (
            ("1e-3", "1e-1"),
            Float(1e-3, 1e-1),
        ),
        (
            (1e-5, 1e-1),
            Float(1e-5, 1e-1),
        ),
        (
            {"type": "float", "lower": 0.00001, "upper": "1e-1", "log": True},
            Float(0.00001, 0.1, log=True),
        ),
        (
            {"type": "int", "lower": 3, "upper": 30, "is_fidelity": True},
            Integer(3, 30, is_fidelity=True),
        ),
        (
            {
                "type": "int",
                "lower": "1e2",
                "upper": "3E4",
                "log": True,
                "is_fidelity": False,
            },
            Integer(100, 30000, log=True, is_fidelity=False),
        ),
        (
            {"type": "float", "lower": "3.3e-5", "upper": "1.5E-1"},
            Float(3.3e-5, 1.5e-1),
        ),
        (
            {"type": "cat", "choices": [2, "sgd", "10e-3"]},
            Categorical([2, "sgd", 0.01]),
        ),
        (
            0.5,
            Constant(0.5),
        ),
        (
            "1e3",
            Constant(1000),
        ),
        (
            {"type": "cat", "choices": ["adam", "sgd", "rmsprop"]},
            Categorical(["adam", "sgd", "rmsprop"]),
        ),
        (
            {
                "lower": 0.00001,
                "upper": 0.1,
                "log": True,
                "prior": 3.3e-2,
                "prior_confidence": "high",
            },
            Float(0.00001, 0.1, log=True, prior=3.3e-2, prior_confidence="high"),
        ),
    ],
)
def test_type_deduction_succeeds(config: Any, expected: Parameter) -> None:
    parameter = parsing.as_parameter(config)
    assert parameter == expected


@pytest.mark.parametrize(
    "config",
    [
        {"type": int, "lower": 0.00001, "upper": 0.1, "log": True},  # Invalid type
        (1, 2.5),  # int and float
        (1, 2, 3),  # too many values
        (1,),  # too few values
    ],
)
def test_parsing_fails(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parsing.as_parameter(config)
