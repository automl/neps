from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from extrapolaw.state.pipeline_eval import EvaluatePipelineReturn, UserResult


def d(obj: Any = None, cost: Any = None, lc: Any = None) -> dict:
    return {
        "default_objective_to_minimize_value": obj,
        "default_cost_value": cost,
        "default_learning_curve": lc,
    }


# Since exception equality is identity matching, we need to have a single one.
EXCEPTION = Exception("")


@pytest.mark.parametrize(
    ("inp", "defaults", "expected"),
    [
        # Exception
        (
            EXCEPTION,
            d(obj=1.0, cost=2.0, lc=[5.0]),
            UserResult(
                objective_to_minimize=1.0,
                cost=2.0,
                learning_curve=[5.0],
                exception=EXCEPTION,
            ),
        ),
        (
            EXCEPTION,
            d(obj=1.0, cost=2.0, lc="objective_to_minimize"),
            UserResult(
                objective_to_minimize=1.0,
                cost=2.0,
                learning_curve=[1.0],
                exception=EXCEPTION,
            ),
        ),
        # Mappings
        (
            {"objective_to_minimize": 1.0, "cost": 2.0},
            d(),
            UserResult(objective_to_minimize=1.0, cost=2.0),
        ),
        (
            {"objective_to_minimize": np.float64(4.5), "cost": 2},
            d(),
            UserResult(objective_to_minimize=4.5, cost=2.0),
        ),
        (
            {"objective_to_minimize": float("inf"), "cost": 2.0},
            d(),
            UserResult(objective_to_minimize=float("inf"), cost=2.0),
        ),
        (
            {"objective_to_minimize": 1.0},
            d(cost=2.0, lc=[5.0]),
            UserResult(objective_to_minimize=1.0, cost=2.0, learning_curve=[5.0]),
        ),
        (
            {"objective_to_minimize": 1.0, "cost": 2.0},
            d(lc="objective_to_minimize"),
            UserResult(objective_to_minimize=1.0, cost=2.0, learning_curve=[1.0]),
        ),
        # Exception
        (
            {"exception": EXCEPTION},
            d(obj=1.0, cost=2.0, lc=[5.0]),
            UserResult(
                objective_to_minimize=1.0,
                cost=2.0,
                learning_curve=[5.0],
                exception=EXCEPTION,
            ),
        ),
        (
            {"exception": EXCEPTION},
            d(obj=1.0, cost=2.0, lc="objective_to_minimize"),
            UserResult(
                objective_to_minimize=1.0,
                cost=2.0,
                learning_curve=[1.0],
                exception=EXCEPTION,
            ),
        ),
        (
            {
                "objective_to_minimize": 1.0,
                "cost": 2.0,
                "learning_curve": [5.0],
                "exception": EXCEPTION,
            },
            d(),
            UserResult(
                objective_to_minimize=1.0,
                cost=2.0,
                learning_curve=[5.0],
                exception=EXCEPTION,
            ),
        ),
        (
            {
                "objective_to_minimize": 1.0,
                "cost": 2.0,
                "info_dict": {"a": "b"},
            },
            d(),
            UserResult(objective_to_minimize=1.0, cost=2.0, extra={"a": "b"}),
        ),
        (
            {
                "objective_to_minimize": 1.0,
                "cost": 2.0,
                "inference_cost": 2,
                "n_params": np.int64(1000),
                "n_data": 5e6,
                "downstream_performances": {"task_acc": np.float32(0.5)},
            },
            d(),
            UserResult(
                objective_to_minimize=1.0,
                cost=2.0,
                inference_cost=2.0,
                n_params=1000.0,
                n_data=5e6,
                downstream_performances={"task_acc": 0.5},
            ),
        ),
    ],
)
def test_user_result_parse_success(
    inp: EvaluatePipelineReturn,
    defaults: dict,
    expected: UserResult,
) -> None:
    parsed = UserResult.parse(inp, **defaults)
    assert parsed == expected


# Most errors really just revolve around invalid input with a dict.
@pytest.mark.parametrize(
    ("inp", "defaults"),
    [
        # Not a dictionary or an exception
        (object(), d()),
        (1.0, d()),
        (np.float64(4.5), d(cost=2.0)),
        ([1.0, 2.0], d(cost=2.0)),
        ("1.0", d()),
        # No cost, though successful
        ({"objective_to_minimize": 1.0}, d()),
        ({"objective_to_minimize": 1.0, "cost": None}, d()),
        # Bad values
        ({"objective_to_minimize": object()}, d()),
        # Multi-objective is not supported
        ({"objective_to_minimize": [1.0, 2.0], "cost": 2.0}, d()),
        ({"objective_to_minimize": 1.0, "cost": object()}, d()),
        ({"objective_to_minimize": 1.0, "learning_curve": object()}, d()),
        ({"objective_to_minimize": 1.0, "info_dict": object()}, d()),
        ({"objective_to_minimize": 1.0, "exception": object()}, d()),
        ({"objective_to_minimize": 1.0, "cost": 2.0, "inference_cost": "2"}, d()),
        ({"objective_to_minimize": 1.0, "cost": 2.0, "n_params": "1e3"}, d()),
        ({"objective_to_minimize": 1.0, "cost": 2.0, "n_data": [5]}, d()),
        ({"objective_to_minimize": 1.0, "downstream_performances": [0.5]}, d()),
        ({"objective_to_minimize": 1.0, "downstream_performances": {"a": "b"}}, d()),
        # Bad defaults (obj)
        (EXCEPTION, d(obj=object())),
        ({"exception": EXCEPTION}, d(obj=object())),
        # Bad defaults (cost)
        (EXCEPTION, d(cost=object())),
        ({"objective_to_minimize": 1.0}, d(cost=object())),
        # Bad defaults (lc)
        (EXCEPTION, d(lc=object())),
        ({"objective_to_minimize": 1.0, "cost": 2.0}, d(lc=object())),
        # No objective_to_minimize or no exception
        ({"cost": 2.0}, d()),
        # Catch legacy
        ({"objective_to_minimize": 1.0, "info_dict": {"learning_curve": None}}, d()),
    ],
)
def test_user_result_parse_errors(
    inp: EvaluatePipelineReturn,
    defaults: dict,
) -> None:
    with pytest.raises(ValueError):
        UserResult.parse(inp, **defaults)
