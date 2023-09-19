from __future__ import annotations

from typing import Any


def get_loss(
    result: str | dict | float,
    loss_value_on_error: float | None = None,
    ignore_errors: bool = False,
) -> float | Any:
    if result == "error":
        if ignore_errors:
            return "error"
        elif loss_value_on_error is None:
            raise ValueError(
                "An error happened during the execution of your run_pipeline function."
                " You have three options: 1. If the error is expected and corresponds to"
                " a loss value in your application (e.g., 0% accuracy), you can set"
                " loss_value_on_error to some float. 2. If sometimes your pipeline"
                " crashes randomly, you can set ignore_errors=True. 3. Fix your error."
            )
        else:
            return loss_value_on_error
    elif isinstance(result, dict):
        return float(result["loss"])
    else:
        return float(result)


def get_cost(
    result: str | dict | float,
    cost_value_on_error: float | None = None,
    ignore_errors: bool = False,
) -> float | Any:
    if result == "error":
        if ignore_errors:
            return "error"
        elif cost_value_on_error is None:
            raise ValueError(
                "An error happened during the execution of your run_pipeline function."
                " You have three options: 1. If the error is expected and corresponds to"
                " a cost value in your application, you can set"
                " cost_value_on_error to some float. 2. If sometimes your pipeline"
                " crashes randomly, you can set ignore_errors=True. 3. Fix your error."
            )
        else:
            return cost_value_on_error
    elif isinstance(result, dict):
        return float(result["cost"])
    else:
        return float(result)


def get_learning_curve(
    result: str | dict | float,
    learning_curve_on_error: list[float] | float | None = None,
    ignore_errors: bool = False,
) -> list[float] | Any:
    if result == "error":
        if ignore_errors:
            return "error"
        elif learning_curve_on_error is None:
            raise ValueError(
                "An error happened during the execution of your run_pipeline function."
                " You have three options: 1. If the error is expected and corresponds to"
                " a learning curve value in your application, you can set"
                " learning_curve_on_error to some float or list of floats."
                " 2. If sometimes your pipeline"
                " crashes randomly, you can set ignore_errors=True. 3. Fix your error."
            )
        elif isinstance(learning_curve_on_error, float):
            learning_curve_on_error = [learning_curve_on_error]
        return learning_curve_on_error
    elif isinstance(result, dict):
        return result["learning_curve"]
    else:
        return float(result)
