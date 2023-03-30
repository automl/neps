from __future__ import annotations

from typing import Any, Mapping

from typing_extensions import Literal, TypeAlias

ERROR: Literal["error"] = "error"

Result: TypeAlias = Literal["error"] | str | float | int | Mapping[str, Any]


def get_costlike_key(
    result: Result,
    key,
    error_value: float | None = None,
    ignore_errors: bool = False,
) -> float | Literal["error"]:
    if result == ERROR and error_value is None:
        raise ValueError(
            "An error happened during the execution of your run_pipeline function."
            " You have three options: 1. If the error is expected and corresponds to"
            f" a {key} value in your application (e.g., 0% accuracy), you can set"
            f" {key}_value_on_error to some float. 2. If sometimes your pipeline"
            " crashes randomly, you can set ignore_errors=True. 3. Fix your error."
        )

    if isinstance(result, Mapping) and key not in result:
        raise ValueError(
            f"The result does not contain a '{key}' key. The result is {result}."
        )

    value: str | int | float
    if result == ERROR:
        if ignore_errors:
            return ERROR
        else:
            assert isinstance(error_value, float)
            value = error_value
    elif isinstance(result, Mapping):
        value = result[key]
    else:
        value = result

    # Now that we got some loss, try convert it to float
    if value == ERROR:
        return ERROR

    try:
        return float(value)
    except ValueError as e:
        raise ValueError(
            f"The result is a {type(value)}"
            f" but cannot be converted to a float. The value is {value}."
        ) from e
