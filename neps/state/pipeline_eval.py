"""Encapsulating the call to the users `evaluate_pipeline`."""

from __future__ import annotations

import inspect
import logging
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias
from typing_extensions import TypedDict

import numpy as np

if TYPE_CHECKING:
    from neps.state.settings import DefaultReportValues
    from neps.state.trial import Report, Trial

logger = logging.getLogger(__name__)


class UserResultDict(TypedDict, total=False):
    """The type of things you can include when returning
    a dictionary from `evaluate_pipeline`.

    All fields here are optional. If an exception is provided,
    we count the trial as having failed. Otherwise, we rely
    on at least `objective_to_minimize` to be provided.
    """

    objective_to_minimize: float | Sequence[float] | None
    """The objective value to minimize, single or multi-objective"""

    cost: float | None
    """The cost of the evaluation, if any"""

    exception: Exception | None
    """The exception that occured."""

    learning_curve: Sequence[float] | Sequence[Sequence[float]] | None
    """The learning curve for this configuration.

    If a `Sequence[float]` is provided, it is assumed to be a single objective
    learning curve.

    Otherwise a `Sequence[Sequence[float]]` is assumed to be a multiobjective
    learning curve.
    """

    info_dict: Mapping[str, Any] | None
    """Extra information that will be stored with the trials result."""


EvaluatePipelineReturn: TypeAlias = (
    Exception | float | Sequence[float] | UserResultDict | dict
)
"""The type of things `evaluate_pipeline` can return.

* `float`: Interpreted as just the objective value to minimize.
* `Sequence[float]`: Interpreted as multiobjective optimization with these values.
    Order is important.
* `Exception`: The evaluation failed.
* `Mapping[str, Any]`: A dict which follows the layout of
    [`UserResultDict`][neps.state.pipeline_eval.EvaluatePipelineReturn].
"""


@dataclass
class UserResult:
    """The parsed values out of the possibilities the user can return
    from the `evaluate_pipeline_function()`.

    See [`UserResultDict`][neps.state.pipeline_eval.EvaluatePipelineReturn] for the
    possibilities.
    """

    objective_to_minimize: float | list[float] | None = None
    cost: float | None = None
    learning_curve: list[float] | list[list[float]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    exception: Exception | None = None

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        # This is the type that we will use to profliferate to the rest of NePS.
        # It's super important that everything that we have as types here is correct.
        # The rest of NePS has type-safety built in and expects these to be correct.
        # Hence we do this extra explicit check incase the parsing function manages
        # to miss something in the conversion to this UserResult.
        if self.exception is None and self.objective_to_minimize is None:
            raise ValueError(
                "Either an exception or an objective value to minimize must be provided,"
                f" but got `None` from the parsed result of {self}."
            )

        try:
            # We just have to do some annoying things to ensure everything here
            # is then serializable.
            match self.objective_to_minimize:
                case Sequence():
                    self.objective_to_minimize = [
                        float(v) for v in self.objective_to_minimize
                    ]
                case np.number() | float() | int():
                    self.objective_to_minimize = float(self.objective_to_minimize)
                case None:
                    pass
                case _:
                    raise ValueError(
                        "The 'objective_to_minimize' should be either a float,"
                        f" a sequence of floats or None. Got {self.objective_to_minimize}"
                    )

            match self.cost:
                case np.number() | float() | int():
                    self.cost = float(self.cost)
                case None:
                    pass
                case _:
                    raise ValueError(
                        f"The 'cost' should be either a float or None. Got {self.cost}"
                    )

            match self.learning_curve:
                # Multi-objective
                case Sequence() if isinstance(self.learning_curve[0], Sequence):
                    self.learning_curve = [
                        [float(v) for v in moo]  # type: ignore
                        for moo in self.learning_curve
                    ]
                case Sequence():
                    self.learning_curve = [float(v) for v in self.learning_curve]  # type: ignore
                case None:
                    pass
                case _:
                    raise ValueError(
                        "The 'learning_curve' should be either a sequence of floats,"
                        f" a sequence of sequences of floats or None."
                        f" Got {self.learning_curve}"
                    )

            match self.exception:
                case None | Exception():
                    pass
                case _:
                    raise ValueError(
                        "The 'exception' should be an exception or None,"
                        f" but got {self.exception}"
                    )

            match self.extra:
                case Mapping():
                    self.extra = dict(self.extra)
                case _:
                    raise ValueError(
                        f"The 'extra' should be a dictionary. Got {self.extra}"
                    )

        except Exception as e:
            raise ValueError(f"Failed to parse the result {self}.") from e

    # This function is damn annoying as there's many ways to return things in NePS
    # I'm sorry for the type signature but we can parse at many possibilities and
    # provide defaults for multi-objective, single objective and such...
    # The mess her is for parsing so that the rest of the code base can rely on it
    # being clean.
    @classmethod
    def parse(  # noqa: C901, PLR0912, PLR0915
        cls,
        user_result: EvaluatePipelineReturn,
        *,
        default_objective_to_minimize_value: float | list[float] | None,
        default_cost_value: float | None = None,
        default_learning_curve: (
            Literal["objective_to_minimize"]
            | list[float]  # Single obj curve
            | list[list[float]]  # Multi obj curve
            | None
        ) = None,
    ) -> UserResult:
        """Parse the return type a user can provide from `evaluate_pipeline`."""
        objective_to_minimize: float | list[float] | None
        cost: float | None
        learning_curve: list[float] | list[list[float]] | None
        extra_info: dict[str, Any]
        exception: Exception | None

        match user_result:
            # Start easy, single objective result only
            case int() | float() | np.number():
                match default_learning_curve:
                    case "objective_to_minimize":  # Take the obj_to_minimize as the curve
                        learning_curve = [float(user_result)]
                    case None | Sequence():  # Go with the default (list or None)
                        learning_curve = default_learning_curve  # type: ignore
                    case _:
                        raise ValueError(
                            "The default learning curve should be either None or"
                            f" 'objective_to_minimize'. Got {default_learning_curve}"
                        )

                return UserResult(
                    objective_to_minimize=user_result,
                    learning_curve=learning_curve,
                    cost=default_cost_value,
                    exception=None,
                    extra={},
                )

            # Multiobjective result
            case Sequence():
                if not all(isinstance(v, float | int | np.number) for v in user_result):
                    val_types = ", ".join(f"{v}: {type(v)}" for v in user_result)
                    raise ValueError(
                        "All values in the multiobjective result should be floats,"
                        f" but got {val_types}"
                    )
                objective_to_minimize = [float(v) for v in user_result]

                match default_learning_curve:
                    case "objective_to_minimize":
                        learning_curve = [objective_to_minimize]
                    case None | Sequence():
                        learning_curve = default_learning_curve  # type: ignore
                    case _:
                        raise ValueError(
                            "The default learning curve should be either None or"
                            f" 'objective_to_minimize'. Got {default_learning_curve}"
                        )

                return UserResult(
                    objective_to_minimize=objective_to_minimize,
                    extra={},
                    cost=default_cost_value,
                    learning_curve=learning_curve,
                    exception=None,
                )

            # An Error
            case Exception():
                objective_to_minimize = default_objective_to_minimize_value
                match default_learning_curve:
                    case "objective_to_minimize":
                        learning_curve = (  # obj_to_minimize as the curve, if any
                            [objective_to_minimize]  # type: ignore
                            if objective_to_minimize is not None
                            else None
                        )
                    case None | Sequence():
                        learning_curve = default_learning_curve  # type: ignore
                    case _:
                        raise ValueError(
                            "The default learning curve should be either None or"
                            f" 'objective_to_minimize'. Got {default_learning_curve}"
                        )

                return UserResult(
                    objective_to_minimize=default_objective_to_minimize_value,
                    cost=default_cost_value,
                    learning_curve=learning_curve,
                    exception=user_result,
                    extra={},
                )

            # A UserResultDict, the most annoying to parse out as theoretically
            # the could provide everything)
            case Mapping():
                _result = dict(user_result)

                popped_exception = _result.pop("exception", None)
                if (
                    not isinstance(popped_exception, Exception)
                    and popped_exception is not None
                ):
                    raise ValueError(
                        "The 'exception' should be an exception or None,"
                        f" but got {popped_exception}"
                    )
                exception = popped_exception

                popped_obj = _result.pop(
                    "objective_to_minimize", default_objective_to_minimize_value
                )
                match popped_obj:
                    case None:
                        objective_to_minimize = popped_obj
                    case int() | float() | np.number():
                        objective_to_minimize = float(popped_obj)
                    case Sequence():
                        objective_to_minimize = [float(v) for v in popped_obj]
                    case _:
                        raise ValueError(
                            "The 'objective_to_minimize' should be either a float,"
                            f" a sequence of floats or None. Got {popped_obj}"
                        )

                popped_cost = _result.pop("cost", default_cost_value)
                match popped_cost:
                    case None:
                        cost = popped_cost
                    case int() | float() | np.number():
                        cost = float(popped_cost)
                    case _:
                        raise ValueError(
                            f"The 'cost' should be either a float or None."
                            f" Got {popped_cost}"
                        )

                # Learning curve is annoying as we have a cross product of possible
                # learning curve returns and what to do if there is a default.
                popped_curve = _result.pop("learning_curve", None)
                match popped_curve:
                    case Sequence():
                        # Easiest case, just use it and assume it's correctly shaped.
                        # TODO: Could check the learning curve is 2d if multiobjective
                        # objective # was provided.
                        learning_curve = list(popped_curve)
                    case None:
                        # If no learning curve, see what to do with the default
                        match default_learning_curve:
                            case None:
                                learning_curve = None
                            case "objective_to_minimize":
                                learning_curve = (
                                    [objective_to_minimize]  # type: ignore
                                    if objective_to_minimize is not None
                                    else None
                                )
                            case Sequence():
                                learning_curve = default_learning_curve  # type: ignore
                            case _:
                                raise ValueError(
                                    "The default learning curve should be either None, "
                                    " 'objective_to_minimize' or sequence."
                                    f" Got {default_learning_curve}"
                                )
                    case _:
                        raise ValueError(
                            "The 'learning_curve' should be either a sequence of floats,"
                            f" a sequence of sequences of floats or None."
                            f" Got {popped_curve}"
                        )

                popped_extra_info = _result.pop("info_dict", {})
                if not isinstance(popped_extra_info, Mapping):
                    raise ValueError(
                        "The 'info_dict' should be a dictionary, but got"
                        f" {popped_extra_info}"
                    )
                extra_info = dict(popped_extra_info)

                # Legacy
                if "learning_curve" in extra_info:
                    raise ValueError(
                        "Please provide 'learning_curve' in the top level of the"
                        " dictionary, not in 'info_dict'."
                    )

                return UserResult(
                    objective_to_minimize=objective_to_minimize,
                    cost=cost,
                    extra=extra_info,
                    learning_curve=learning_curve,
                    exception=exception,
                )

            case _:
                raise ValueError(
                    "The user result should be either a float, a sequence of floats,"
                    f" an exception or a dictionary. Got {user_result}"
                )


def _eval_trial(
    *,
    trial: Trial,
    default_report_values: DefaultReportValues,
    fn: Callable[..., Any],
    **kwargs: Any,
) -> Report:
    start = time.monotonic()
    try:
        user_result = fn(**kwargs, **trial.config)
    # Something went wrong in evaluation
    except Exception as e:
        duration = time.monotonic() - start
        time_end = time.time()
        logger.error(f"Error during evaluation of '{trial.id}': {trial.config}.")
        logger.exception(e)
        report = trial.set_complete(
            report_as="crashed",
            objective_to_minimize=default_report_values.objective_value_on_error,
            cost=default_report_values.cost_value_on_error,
            learning_curve=default_report_values.learning_curve_on_error,
            extra=None,
            err=e,
            tb=traceback.format_exc(),
            time_end=time_end,
            evaluation_duration=duration,
        )
    else:
        duration = time.monotonic() - start
        time_end = time.time()
        match user_result:
            case dict():
                filtered_data = {k: v for k, v in user_result.items() if k != "info_dict"}
                logger.info(f"Successful evaluation of '{trial.id}': {filtered_data}.")
            case _:  # TODO: Revisit this and check all possible cases
                logger.info(f"Successful evaluation of '{trial.id}': {user_result}.")

        result = UserResult.parse(
            user_result,
            default_cost_value=default_report_values.cost_if_not_provided,
            default_objective_to_minimize_value=default_report_values.objective_value_on_error,
            default_learning_curve=default_report_values.learning_curve_if_not_provided,
        )
        report = trial.set_complete(
            report_as="success",
            objective_to_minimize=result.objective_to_minimize,
            cost=result.cost,
            learning_curve=result.learning_curve,
            err=result.exception,
            tb=None,
            extra=result.extra,
            time_end=time_end,
            evaluation_duration=duration,
        )

    return report


def evaluate_trial(
    trial: Trial,
    *,
    evaluation_fn: Callable[..., Any],
    default_report_values: DefaultReportValues,
) -> tuple[Trial, Report]:
    """Evaluates a trial from a user and parses the results into a `Report`."""
    trial_location = Path(trial.metadata.location)
    prev_trial_location = (
        Path(trial.metadata.previous_trial_location)
        if trial.metadata.previous_trial_location is not None
        else None
    )

    params = {
        "pipeline_directory": trial_location,
        "previous_pipeline_directory": prev_trial_location,
    }
    sigkeys = inspect.signature(evaluation_fn).parameters.keys()
    injectable_params = {key: val for key, val in params.items() if key in sigkeys}
    report = _eval_trial(
        trial=trial,
        fn=evaluation_fn,
        default_report_values=default_report_values,
        **injectable_params,
    )
    return trial, report
