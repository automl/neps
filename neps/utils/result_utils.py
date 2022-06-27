from __future__ import annotations

from typing import Any


# loss_value_on_error defaults to inf in order not be break existing functionality.
def get_loss(
    result: str | dict | float, loss_value_on_error: float = None
) -> float | Any:
    if result == "error":
        if loss_value_on_error is None:
            raise ValueError(
                "An error happened and loss_value_on_error is not provided,"
                " please fix the error or provide a float to ignore the "
                "error"
            )
        return loss_value_on_error
    elif isinstance(result, dict):
        return float(result["loss"])
    else:
        return float(result)


# cost_value_on_error defaults to inf in order not be break existing functionality.
def get_cost(
    result: str | dict | float, cost_value_on_error: float = None
) -> float | Any:
    if result == "error":
        if cost_value_on_error is None:
            raise ValueError(
                "An error happened and cost_value_on_error is not provided,"
                " please fix the error or provide a float to ignore the "
                "error"
            )
        return cost_value_on_error
    elif isinstance(result, dict):
        return float(result["cost"])
    else:
        return float(result)
