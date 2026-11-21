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
    from extrapolaw.state.settings import DefaultReportValues
    from extrapolaw.state.trial import Report, Trial

logger = logging.getLogger(__name__)


class UserResultDict(TypedDict, total=False):
    """The type of things you can include when returning
    a dictionary from `evaluate_pipeline`.

    `objective_to_minimize` and `cost` are required, the other fields are optional.
    If an exception is provided, we count the trial as having failed.
    """

    objective_to_minimize: float | None
    """The objective value to minimize, e.g. the loss."""

    cost: float | None
    """The training cost of the evaluation, e.g. its FLOPs. Required, unless a
    `cost_estimator` is passed to `extrapolaw.acquire()`."""

    inference_cost: float | None
    """The cost of using the trained model once, e.g. the FLOPs of one forward pass.
    The scaling plots are also drawn over it."""

    n_params: float | None
    """The number of parameters of the model."""

    n_data: float | None
    """The amount of training data, e.g. the number of tokens."""

    downstream_performances: Mapping[str, float] | None
    """The model's performance on downstream tasks, by name, e.g.
    `{"hellaswag_acc": 0.41}`. The scaling plots draw each of them over the cost."""

    exception: Exception | None
    """The exception that occured."""

    learning_curve: Sequence[float] | None
    """The learning curve for this configuration."""

    info_dict: Mapping[str, Any] | None
    """Extra information that will be stored with the trials result."""


EvaluatePipelineReturn: TypeAlias = Exception | UserResultDict | dict
"""The type of things `evaluate_pipeline` can return.

* `Exception`: The evaluation failed.
* `Mapping[str, Any]`: A dict which follows the layout of
    [`UserResultDict`][extrapolaw.state.pipeline_eval.EvaluatePipelineReturn].
"""


@dataclass
class UserResult:
    """The parsed values out of the possibilities the user can return
    from the `evaluate_pipeline_function()`.

    See [`UserResultDict`][extrapolaw.state.pipeline_eval.EvaluatePipelineReturn] for the
    possibilities.
    """

    objective_to_minimize: float | None = None
    cost: float | None = None
    learning_curve: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    exception: Exception | None = None
    inference_cost: float | None = None
    downstream_performances: dict[str, float] = field(default_factory=dict)
    n_params: float | None = None
    n_data: float | None = None

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        # This is the type that we will use to profliferate to the rest of Extrapolaw.
        # It's super important that everything that we have as types here is correct.
        # The rest of Extrapolaw has type-safety built in and expects these to be correct.
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
                case np.number() | float() | int():
                    self.objective_to_minimize = float(self.objective_to_minimize)
                case None:
                    pass
                case _:
                    raise ValueError(
                        "The 'objective_to_minimize' should be either a float or None."
                        f" Got {self.objective_to_minimize}"
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
                case Sequence():
                    self.learning_curve = [float(v) for v in self.learning_curve]
                case None:
                    pass
                case _:
                    raise ValueError(
                        "The 'learning_curve' should be either a sequence of floats or"
                        f" None. Got {self.learning_curve}"
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
            match self.inference_cost:
                case np.number() | float() | int():
                    self.inference_cost = float(self.inference_cost)
                case None:
                    pass
                case _:
                    raise ValueError(
                        "The 'inference_cost' should be either a float or None."
                        f" Got {self.inference_cost}"
                    )

            match self.downstream_performances:
                case Mapping():
                    self.downstream_performances = {
                        str(name): float(value)
                        for name, value in self.downstream_performances.items()
                    }
                case _:
                    raise ValueError(
                        "The 'downstream_performances' should be a dictionary of"
                        f" floats. Got {self.downstream_performances}"
                    )

            match self.n_params:
                case np.number() | float() | int():
                    self.n_params = float(self.n_params)
                case None:
                    pass
                case _:
                    raise ValueError(
                        "The 'n_params' should be either a float or None."
                        f" Got {self.n_params}"
                    )

            match self.n_data:
                case np.number() | float() | int():
                    self.n_data = float(self.n_data)
                case None:
                    pass
                case _:
                    raise ValueError(
                        "The 'n_data' should be either a float or None."
                        f" Got {self.n_data}"
                    )

        except Exception as e:
            raise ValueError(f"Failed to parse the result {self}.") from e

    # This function is damn annoying as there's many ways to return things in Extrapolaw
    # The mess her is for parsing so that the rest of the code base can rely on it
    # being clean.
    @classmethod
    def parse(  # noqa: C901, PLR0912, PLR0915
        cls,
        user_result: EvaluatePipelineReturn,
        *,
        default_objective_to_minimize_value: float | None,
        default_cost_value: float | None = None,
        default_learning_curve: Literal["objective_to_minimize"]
        | list[float]
        | None = None,
    ) -> UserResult:
        """Parse the return type a user can provide from `evaluate_pipeline`."""
        objective_to_minimize: float | None
        cost: float | None
        learning_curve: list[float] | None
        extra_info: dict[str, Any]
        exception: Exception | None

        match user_result:
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
                        raise ValueError(
                            "Multi-objective optimization is not supported, the"
                            " 'objective_to_minimize' should be a single float."
                            f" Got {popped_obj}"
                        )
                    case _:
                        raise ValueError(
                            "The 'objective_to_minimize' should be either a float or"
                            f" None. Got {popped_obj}"
                        )

                popped_cost = _result.pop("cost", default_cost_value)
                match popped_cost:
                    case None:
                        cost = popped_cost
                    case int() | float() | np.number():
                        cost = float(popped_cost)
                    case _:
                        raise ValueError(
                            "The 'cost' should be either a float or None."
                            f" Got {popped_cost}"
                        )
                # A successful evaluation needs its training cost
                if exception is None and cost is None:
                    raise ValueError(
                        "The result has no 'cost'. Return the training cost, e.g. its"
                        " FLOPs, or pass a `cost_estimator` to `extrapolaw.acquire()`."
                    )

                # Learning curve is annoying as we have a cross product of possible
                # learning curve returns and what to do if there is a default.
                popped_curve = _result.pop("learning_curve", None)
                match popped_curve:
                    case Sequence():
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
                            "The 'learning_curve' should be either a sequence of floats"
                            f" or None. Got {popped_curve}"
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
                    inference_cost=_result.pop("inference_cost", None),
                    downstream_performances=_result.pop("downstream_performances", None)
                    or {},
                    n_params=_result.pop("n_params", None),
                    n_data=_result.pop("n_data", None),
                )

            case _:
                raise ValueError(
                    "The result should be a dictionary with at least"
                    " 'objective_to_minimize' and 'cost', or an exception."
                    f" Got {user_result}"
                )


def _eval_trial(
    *,
    trial: Trial,
    default_report_values: DefaultReportValues,
    fn: Callable[..., Any],
    **kwargs: Any,
) -> Report | None:
    start = time.monotonic()
    try:
        user_result = fn(**kwargs, **trial.config)
        if user_result is None:
            return None
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
                filtered_data = {
                    k: v
                    for k, v in user_result.items()
                    if k not in ["info_dict", "learning_curve"]
                }
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
            inference_cost=result.inference_cost,
            downstream_performances=result.downstream_performances,
            n_params=result.n_params,
            n_data=result.n_data,
        )

    return report


def evaluate_trial(
    trial: Trial,
    *,
    evaluation_fn: Callable[..., Any],
    default_report_values: DefaultReportValues,
) -> tuple[Trial, Report | None]:
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
        "pipeline_id": trial.id,
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
