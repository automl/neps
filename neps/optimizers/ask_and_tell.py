"""Implements a basic Ask-and-Tell wrapper around an optimizer.

This is a simple wrapper around an optimizer that allows you to ask for new
configurations and report the results back to the optimizer, without
using the NePS runtime to run the evaluation for you.

This puts you in charge of getting new configurations,
evaluating the configuration and reporting back the results, in whatever
order you would prefer. For example, you could
[`ask()`][neps.optimizers.ask_and_tell.AskAndTell.ask] twice to get two configuration,
evaluate both configurations in parallel, and then
[`tell()`][neps.optimizers.ask_and_tell.AskAndTell.tell] results back to the optimizer.

```python
from neps import AskAndTell

# Wrap an optimizer
my_optimizer = AskAndTell(MyOptimizer(space, ...))

# Ask for a new configuration
trial = my_optimizer.ask()

# The things you would normally get into `evaluate_pipeline`
config_id = trial.config_id
config = trial.config
previous_config_id = trial.metadata.previous_trial_id
previous_trial_path = trial.metadata.previous_trial_location

# Evaluate the configuration
loss = evaluate(config)

# Tell the optimizer the result
my_optimizer.tell(config_id, loss)
```

Importantly, we expose a little more of the information that is normally
hidden from you by exposing the [`Trial`][neps.state.trial.Trial] object.
This carries most of the meta-information that is normally written to disk
and stored with each evaluation.

---

You can also report your own custom configurations, for example to warmstart
an optimizer with previous results:

```python
optimizer.tell_custom(
    config_id="my_config_id",  # Make sure to give it a unique id
    config={"a": 1, "b": 2},
    result={"objective_to_minimize": 0.5},  # The same as the return evaluate_pipeline
)
```

You can provide a lot more info that normally the neps runtime would fill int
for you. For a full list, please see
[`tell_custom`][neps.optimizers.ask_and_tell.AskAndTell.tell_custom].

---

Please see [`AskFunction`][neps.optimizers.optimizer.AskFunction] for more information
on how to implement your own optimizer.
"""

from __future__ import annotations

import datetime
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

from neps.optimizers.optimizer import AskFunction, SampledConfig
from neps.state import EvaluatePipelineReturn, Trial, UserResult

if TYPE_CHECKING:
    from neps.state.optimizer import BudgetInfo
    from neps.state.pipeline_eval import EvaluatePipelineReturn


def _default_worker_name() -> str:
    isoformat = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return f"{os.getpid()}-{isoformat}"


@dataclass
class AskAndTell:
    """A wrapper around an optimizer that allows you to ask for new configurations."""

    optimizer: AskFunction
    """The optimizer to wrap."""

    worker_id: str = field(default_factory=_default_worker_name)
    """The worker id used to fill out the trial metadata."""

    trials: dict[str, Trial] = field(init=False, default_factory=dict)
    """The trials that the optimizer is aware of, whether sampled or with a result."""

    @overload
    def ask(
        self,
        *,
        n: int,
        budget_info: BudgetInfo | None = ...,
    ) -> list[Trial]: ...

    @overload
    def ask(
        self,
        *,
        n: None = None,
        budget_info: BudgetInfo | None = ...,
    ) -> Trial: ...

    def ask(
        self,
        *,
        n: int | None = None,
        budget_info: BudgetInfo | None = None,
    ) -> Trial | list[Trial]:
        """Ask the optimizer for a new configuration.

        Args:
            n: The number of configurations to sample at once.
            budget_info: information about the budget constraints. Only
                required if the optimizer needs it. You have the
                responsibility to fill this out, which also allows
                you to handle it more flexibly as you need.

        Returns:
            The sampled trial(s)
        """
        sampled_config = self.optimizer(self.trials, budget_info, n)
        if isinstance(sampled_config, SampledConfig):
            _configs = [sampled_config]
        else:
            _configs = sampled_config

        sample_time = time.time()
        trials: list[Trial] = []
        for sampled in _configs:
            trial = Trial.new(
                trial_id=sampled.id,
                location="",
                config=sampled.config,
                previous_trial=sampled.previous_config_id,
                previous_trial_location="",
                time_sampled=sample_time,
                worker_id=self.worker_id,
            )

            # This is sort of some cruft we have to include here to make
            # it match up with what the runtime would do... oh well
            trial.set_evaluating(
                time_started=sample_time,
                worker_id=self.worker_id,
            )
            self.trials[sampled.id] = trial
            trials.append(trial)

        if n is None:
            return trials[0]

        return trials

    def tell_custom(
        self,
        *,
        config_id: str,
        config: Mapping[str, Any],
        result: EvaluatePipelineReturn,
        time_sampled: float = float("nan"),
        time_started: float = float("nan"),
        time_end: float = float("nan"),
        evaluation_duration: float = float("nan"),
        previous_trial_id: str | None = None,
        worker_id: str | None = None,
        traceback_str: str | None = None,
    ) -> Trial:
        """Report a custom configuration and result to the optimizer.

        This is useful for warmstarting an optimizer with previous results.

        Args:
            config_id: The id of the configuration.
            config: The configuration.
            result: The result of the evaluation. This can be an exception,
                a float, or a mapping of values, similar to that which
                you would return from `evaluate_pipeline` when your normally
                call [`neps.run()`][neps.api.run].
            time_sampled: The time the configuration was sampled.
                Only used as metadata.
            time_started: The time the configuration was started to be evaluated.
                Only used as metadata.
            time_end: The time the configuration was finished being evaluated.
                Only used as metadata.
            evaluation_duration: The duration of the evaluation. Only used
                as metadata
            previous_trial_id: The id of any previous trial that this configuration
                was derived from, for example, the same configuration as an earlier
                one but at a later epoch.
            worker_id: The worker id that sampled this configuration, only to fill in
                metadata if you need.
            traceback_str: The traceback of any error, only to fill in
                metadata if you need.

        Returns:
            The trial object that was created. You can find the report
            generated at `trial.report`. You do not require this at any
            other point and the return value can safely be ignored if you wish.
        """
        if config_id in self.trials:
            raise ValueError(f"Config id '{config_id}' already exists!")

        if worker_id is None:
            worker_id = self.worker_id

        parsed_result = UserResult.parse(
            result,
            default_objective_to_minimize_value=None,
            default_cost_value=None,
            default_learning_curve=None,
        )
        report_as: Literal["success", "crashed"] = (
            "success" if parsed_result.exception is None else "crashed"
        )

        # Just go through the motions of the trial life-cycle
        trial = Trial.new(
            trial_id=config_id,
            location="",
            config=config,
            previous_trial=previous_trial_id,
            previous_trial_location="",
            time_sampled=time_sampled,
            worker_id=worker_id,
        )
        trial.set_evaluating(
            time_started=time_started,
            worker_id=worker_id,
        )
        trial.set_complete(
            report_as=report_as,
            objective_to_minimize=parsed_result.objective_to_minimize,
            cost=parsed_result.cost,
            learning_curve=parsed_result.learning_curve,
            extra=parsed_result.extra,
            err=parsed_result.exception,
            time_end=time_end,
            evaluation_duration=evaluation_duration,
            tb=traceback_str,
        )
        self.trials[config_id] = trial
        return trial

    def tell(
        self,
        trial: str | Trial,
        result: EvaluatePipelineReturn,
        *,
        time_end: float | None = None,
        evaluation_duration: float | None = None,
        traceback_str: str | None = None,
    ) -> Trial:
        """Report the result of an evaluation back to the optimizer.

        Args:
            config_id: The id of the configuration you got from
                [`ask()`][neps.optimizers.ask_and_tell.AskAndTell.ask].
            result: The result of the evaluation. This can be an exception,
                a float, or a mapping of values, similar to that which
                you would return from `evaluate_pipeline` when your normally
                call [`neps.run()`][neps.api.run].
            time_end: The time the configuration was finished being evaluated.
                Defaults to `time.time()`. Only used as metadata.
            evaluation_duration: The duration of the evaluation. Defaults
                to the difference between when it was
                [`ask()`][neps.optimizers.ask_and_tell.AskAndTell.ask]ed
                for and now. Only used as metadata
            traceback_str: The traceback of any error, only to fill in
                metadata if you need.

        Returns:
            The trial object that was updated. You can find the report
            generated at `trial.report`. You do not require this at any
            other point and the return value can safely be ignored if you wish.
        """
        trial_id = trial if isinstance(trial, str) else trial.id

        _trial = self.trials.get(trial_id)
        if _trial is None:
            raise ValueError(
                f"Unknown trial id: {trial_id}."
                f" Known trial ids: {list(self.trials.keys())}"
            )

        parsed_result = UserResult.parse(
            result,
            default_objective_to_minimize_value=None,
            default_cost_value=None,
            default_learning_curve=None,
        )
        report_as: Literal["success", "crashed"] = (
            "success" if parsed_result.exception is None else "crashed"
        )

        _trial = self.trials[_trial.id]
        _trial.set_complete(
            report_as=report_as,
            objective_to_minimize=parsed_result.objective_to_minimize,
            cost=parsed_result.cost,
            learning_curve=parsed_result.learning_curve,
            extra=parsed_result.extra,
            time_end=time_end if time_end is not None else time.time(),
            evaluation_duration=evaluation_duration,
            err=parsed_result.exception,
            tb=traceback_str,
        )
        return _trial
