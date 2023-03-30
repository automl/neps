from __future__ import annotations

from typing import Mapping

from typing_extensions import Literal

from metahyper.result import ERROR, get_costlike_key

def get_loss(
    result: str | Mapping | float | int,
    loss_value_on_error: float | None = None,
    ignore_errors: bool = False,
) -> float | Literal["error"]:
    return get_costlike_key(result, "loss", loss_value_on_error, ignore_errors)


def get_cost(
    result: str | dict | float,
    cost_value_on_error: float | None = None,
    ignore_errors: bool = False,
) -> float | Literal["error"]:
    return get_costlike_key(result, "cost", cost_value_on_error, ignore_errors)
