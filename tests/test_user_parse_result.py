from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from neps.state.pipeline_eval import EvaluatePipelineReturn, UserResult


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
        # Float
        (1.0, d(), UserResult(objective_to_minimize=1.0)),
        (np.float64(4.5), d(), UserResult(objective_to_minimize=4.5)),
        (float("inf"), d(), UserResult(objective_to_minimize=float("inf"))),
        (
            1.0,
            d(cost=2.0, lc=[5.0]),
            UserResult(objective_to_minimize=1.0, cost=2.0, learning_curve=[5.0]),
        ),
        (
            1.0,
            d(lc="objective_to_minimize"),
            UserResult(objective_to_minimize=1.0, learning_curve=[1.0]),
        ),
        # Multiobjective
        (
            [1.0, 2.0],
            d(),
            UserResult(objective_to_minimize=[1.0, 2.0]),
        ),
        (
            [1.0, 2.0],
            d(cost=2.0, lc=[5.0, 6.0]),
            UserResult(
                objective_to_minimize=[1.0, 2.0],
                cost=2.0,
                learning_curve=[5.0, 6.0],
            ),
        ),
        (
            [1.0, 2.0],
            d(lc="objective_to_minimize"),
            UserResult(
                objective_to_minimize=[1.0, 2.0],
                learning_curve=[[1.0, 2.0]],
            ),
        ),
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
        # Mappings, the hard ones
        # We first duplicate all the above test in dict form.
        # The we proceed to mapping specific ones
        ({"objective_to_minimize": 1.0}, d(), UserResult(objective_to_minimize=1.0)),
        (
            {"objective_to_minimize": float("inf")},
            d(),
            UserResult(objective_to_minimize=float("inf")),
        ),
        (
            {"objective_to_minimize": 1.0},
            d(cost=2.0, lc=[5.0]),
            UserResult(objective_to_minimize=1.0, cost=2.0, learning_curve=[5.0]),
        ),
        (
            {"objective_to_minimize": 1.0},
            d(lc="objective_to_minimize"),
            UserResult(objective_to_minimize=1.0, learning_curve=[1.0]),
        ),
        # Multiobjective
        (
            {"objective_to_minimize": [1.0, 2.0]},
            d(),
            UserResult(objective_to_minimize=[1.0, 2.0]),
        ),
        (
            {"objective_to_minimize": [1.0, 2.0]},
            d(cost=2.0, lc=[5.0, 6.0]),
            UserResult(
                objective_to_minimize=[1.0, 2.0],
                cost=2.0,
                learning_curve=[5.0, 6.0],
            ),
        ),
        (
            {"objective_to_minimize": [1.0, 2.0]},
            d(lc="objective_to_minimize"),
            UserResult(
                objective_to_minimize=[1.0, 2.0],
                learning_curve=[[1.0, 2.0]],
            ),
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
                "objective_to_minimize": [1.0, 2.0],
                "cost": None,
                "learning_curve": None,
                "exception": None,
            },
            d(lc="objective_to_minimize"),
            UserResult(
                objective_to_minimize=[1.0, 2.0],
                cost=None,
                learning_curve=[[1.0, 2.0]],
                exception=None,
            ),
        ),
        (
            {
                "objective_to_minimize": 1.0,
                "info_dict": {"a": "b"},
            },
            d(),
            UserResult(objective_to_minimize=1.0, extra={"a": "b"}),
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
        # Bad values
        (object(), d()),
        ({"objective_to_minimize": object()}, d()),
        ({"objective_to_minimize": 1.0, "cost": object()}, d()),
        ({"objective_to_minimize": 1.0, "learning_curve": object()}, d()),
        ({"objective_to_minimize": 1.0, "info_dict": object()}, d()),
        ({"objective_to_minimize": 1.0, "exception": object()}, d()),
        # Bad defaults (obj)
        (EXCEPTION, d(obj=object())),
        ({"exception": EXCEPTION}, d(obj=object())),
        # Bad defaults (cost)
        (1.0, d(cost=object())),
        ([1.0, 2.0], d(cost=object())),
        (EXCEPTION, d(cost=object())),
        ({"objective_to_minimize": 1.0}, d(cost=object())),
        # Bad defaults (lc)
        (1.0, d(lc=object())),
        ([1.0, 2.0], d(lc=object())),
        (EXCEPTION, d(lc=object())),
        ({"objective_to_minimize": 1.0}, d(lc=object())),
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
