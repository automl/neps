from __future__ import annotations

from typing import Any


# loss_value_on_error defaults to inf in order not be break existing functionality.
def get_loss(
    result: str | dict | float, loss_value_on_error: float = float("inf")
) -> float | Any:
    if result == "error":
        return loss_value_on_error
    elif isinstance(result, dict):
        return float(result["loss"])
    else:
        return float(result)


# cost_value_on_error defaults to inf in order not be break existing functionality.
def get_cost(
    result: str | dict | float, cost_value_on_error: float = float("inf")
) -> float | Any:
    if result == "error":
        return cost_value_on_error
    elif isinstance(result, dict):
        return float(result["cost"])
    else:
        return float(result)
