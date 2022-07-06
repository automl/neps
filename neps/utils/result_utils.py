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
    result: str | dict | float,
    cost_value_on_error: float = float("inf"),
    default_cost: float = 0,
) -> float | Any:
    if result == "error":
        return cost_value_on_error
    elif isinstance(result, dict):
        return float(result.get("cost", default_cost))
    else:
        return (
            default_cost  # If there is only a float value, it is the loss, not the cost
        )
