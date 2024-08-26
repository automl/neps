"""The main state object that holds all the shared state objects.

This object is used to interact with the shared state objects in a safe atomic
manner, such that each worker can create an identical NePSState and interact with
it without having to worry about locking or out-dated information.

For an actual instantiation of this object, see
[`create_or_load_filebased_neps_state`][neps.state.filebased.create_or_load_filebased_neps_state].
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar, overload

from more_itertools import take

from neps.state.err_dump import ErrDump
from neps.state.optimizer import OptimizationState, OptimizerInfo
from neps.state.trial import Trial

if TYPE_CHECKING:
    from neps.optimizers.base_optimizer import BaseOptimizer
    from neps.state.protocols import Synced, TrialRepo
    from neps.state.seed_snapshot import SeedSnapshot

logger = logging.getLogger(__name__)

# TODO: Technically we don't need the same Location type for all shared objects.
Loc = TypeVar("Loc")
T = TypeVar("T")


@dataclass
class NePSState(Generic[Loc]):
    """The main state object that holds all the shared state objects."""

    location: str

    _trials: TrialRepo[Loc] = field(repr=False)
    _optimizer_info: Synced[OptimizerInfo, Loc]
    _seed_state: Synced[SeedSnapshot, Loc] = field(repr=False)
    _optimizer_state: Synced[OptimizationState, Loc]
    _shared_errors: Synced[ErrDump, Loc] = field(repr=False)

    def put_updated_trial(self, trial: Trial, /) -> None:
        """Update the trial with the new information.

        Args:
            trial: The trial to update.

        Raises:
            VersionMismatchError: If the trial has been updated since it was last
                fetched by the worker using this state. This indicates that some other
                worker has updated the trial in the meantime and the changes from
                this worker are rejected.
        """
        shared_trial = self._trials.get_by_id(trial.id)
        shared_trial.put(trial)

    def get_trial_by_id(self, trial_id: str, /) -> Trial:
        """Get a trial by its id."""
        return self._trials.get_by_id(trial_id).synced()

    def get_trials_by_ids(self, trial_ids: list[str], /) -> dict[str, Trial | None]:
        """Get trials by their ids."""
        return {
            _id: shared_trial.synced()
            for _id, shared_trial in self._trials.get_by_ids(trial_ids).items()
        }

    def sample_trial(
        self,
        optimizer: BaseOptimizer,
        *,
        worker_id: str,
        _sample_hooks: list[Callable] | None = None,
    ) -> Trial:
        """Sample a new trial from the optimizer.

        Args:
            optimizer: The optimizer to sample the trial from.
            worker_id: The worker that is sampling the trial.
            _sample_hooks: A list of hooks to apply to the optimizer before sampling.

        Returns:
            The new trial.
        """
        with (
            self._optimizer_state.acquire() as (
                opt_state,
                put_opt,
            ),
            self._seed_state.acquire() as (seed_state, put_seed_state),
        ):
            trials: dict[Trial.ID, Trial] = {}
            for trial_id, shared_trial in self._trials.all().items():
                trial = shared_trial.synced()
                trials[trial_id] = trial

            seed_state.set_as_global_seed_state()

            # TODO: Not sure if any existing pre_load hooks required
            # it to be done after `load_results`... I hope not.
            if _sample_hooks is not None:
                for hook in _sample_hooks:
                    optimizer = hook(optimizer)

            # NOTE: We don't want optimizers mutating this before serialization
            budget = opt_state.budget.clone() if opt_state.budget is not None else None
            sampled_config, new_opt_state = optimizer.ask(
                trials=trials,
                budget_info=budget,
                optimizer_state=opt_state.shared_state,
            )

            if sampled_config.previous_config_id is not None:
                previous_trial = trials.get(sampled_config.previous_config_id)
                if previous_trial is None:
                    raise ValueError(
                        f"Previous trial '{sampled_config.previous_config_id}' not found."
                    )
                previous_trial_location = previous_trial.metadata.location
            else:
                previous_trial_location = None

            trial = Trial.new(
                trial_id=sampled_config.id,
                location="",  # HACK: This will be set by the `TrialRepo`
                config=sampled_config.config,
                previous_trial=sampled_config.previous_config_id,
                previous_trial_location=previous_trial_location,
                time_sampled=time.time(),
                worker_id=worker_id,
            )
            shared_trial = self._trials.put_new(trial)
            seed_state.recapture()
            put_seed_state(seed_state)
            put_opt(
                OptimizationState(budget=opt_state.budget, shared_state=new_opt_state)
            )

        return trial

    def report_trial_evaluation(
        self,
        trial: Trial,
        report: Trial.Report,
        optimizer: BaseOptimizer,
        *,
        worker_id: str,
    ) -> None:
        """Update the trial with the evaluation report and update the optimizer state
        accordingly.

        Args:
            trial: The trial that was evaluated.
            report: The evaluation report.
            optimizer: The optimizer to update and get the state from
            worker_id: The worker that evaluated the trial.
        """
        shared_trial = self._trials.get_by_id(trial.id)
        # TODO: This would fail if some other worker has already updated the trial.

        # IMPORTANT: We need to attach the report to the trial before updating the things.
        trial.report = report
        shared_trial.put(trial)
        logger.debug("Updated trial '%s' with status '%s'", trial.id, trial.state)
        with self._optimizer_state.acquire() as (opt_state, put_opt_state):
            optimizer.update_state_post_evaluation(opt_state.shared_state, report)

            # TODO: If an optimizer doesn't use the state, this is a waste of time.
            # Update the budget if we have one.
            if opt_state.budget is not None:
                budget_info = opt_state.budget

                if report.cost is not None:
                    budget_info.used_cost_budget += report.cost
            put_opt_state(opt_state)

        if report.err is not None:
            with self._shared_errors.acquire() as (errs, put_errs):
                trial_err = ErrDump.SerializableTrialError(
                    trial_id=trial.id,
                    worker_id=worker_id,
                    err_type=type(report.err).__name__,
                    err=str(report.err),
                    tb=report.tb,
                )
                errs.append(trial_err)
                put_errs(errs)

    def get_errors(self) -> ErrDump:
        """Get all the errors that have occurred during the optimization."""
        return self._shared_errors.synced()

    @overload
    def get_next_pending_trial(self) -> Trial | None: ...
    @overload
    def get_next_pending_trial(self, n: int | None = None) -> list[Trial]: ...

    def get_next_pending_trial(self, n: int | None = None) -> Trial | list[Trial] | None:
        """Get the next pending trial to evaluate.

        Args:
            n: The number of trials to get. If `None`, get the next trial.

        Returns:
            The next trial or a list of trials if `n` is not `None`.
        """
        _pending_itr = (
            shared_trial.synced() for _, shared_trial in self._trials.pending()
        )
        if n is not None:
            return take(n, _pending_itr)
        return next(_pending_itr, None)

    def all_trial_ids(self) -> set[Trial.ID]:
        """Get all the trial ids that are known about."""
        return self._trials.all_trial_ids()

    def get_all_trials(self) -> dict[Trial.ID, Trial]:
        """Get all the trials that are known about."""
        return {_id: trial.synced() for _id, trial in self._trials.all().items()}

    def optimizer_info(self) -> OptimizerInfo:
        """Get the optimizer information."""
        return self._optimizer_info.synced()

    def optimizer_state(self) -> OptimizationState:
        """Get the optimizer state."""
        return self._optimizer_state.synced()
