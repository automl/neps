"""Validation utilities for the NePS package."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from neps.exceptions import NePSError


class DeprecatedArgumentError(NePSError):
    """Raised when a deprecated argument is used."""


def validate_run_pipeline_arguments(f: Callable[..., Any]) -> None:
    """Validate the arguments of a run pipeline function to see if deprcated arguments
    are used.
    """
    evaluation_fn_params = inspect.signature(f).parameters
    if "previous_working_directory" in evaluation_fn_params:
        raise RuntimeError(
            "the argument: 'previous_working_directory' was deprecated. "
            f"In the function: '{f.__name__}', please,  "
            "use 'previous_pipeline_directory' instead. ",
        )
    if "working_directory" in evaluation_fn_params:
        raise RuntimeError(
            "the argument: 'working_directory' was deprecated. "
            f"In the function: '{f.__name__}', please,  "
            "use 'pipeline_directory' instead. ",
        )
