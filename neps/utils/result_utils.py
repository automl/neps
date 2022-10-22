from __future__ import annotations

from typing import Any


def get_loss(
    result: str | dict | float,
    loss_value_on_error: float = None,
    ignore_errors=False,
) -> float | Any:
    if result == "error":
        if ignore_errors:
            return "error"
        elif loss_value_on_error is None:
            raise ValueError(
                "An error happened and loss_value_on_error is not provided. You can"
                " either set loss_value_on_error to some float, set"
                " ignore_errors=True, or fix the error."
            )
        else:
            return loss_value_on_error
    elif isinstance(result, dict):
        return float(result["loss"])
    else:
        return float(result)


def get_cost(
    result: str | dict | float,
    cost_value_on_error: float = None,
    ignore_errors=False,
) -> float | Any:
    if result == "error":
        if ignore_errors:
            return "error"
        elif cost_value_on_error is None:
            raise ValueError(
                "An error happened and loss_value_on_error is not provided. You can"
                " either set loss_value_on_error to some float, set"
                " ignore_errors=True, or fix the error."
            )
        else:
            return cost_value_on_error
    elif isinstance(result, dict):
        return float(result["cost"])
    else:
        return float(result)
