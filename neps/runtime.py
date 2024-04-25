"""Module for the runtime of a single instance of NePS running.

An important advantage of NePS with a running instance per worker and no
multiprocessing is that we can reliably use globals to store information such
as the currently runnig configuraiton, without interfering with other
workers which have launched.

This allows us to have a global `Trial` object which can be accessed
using `import neps.runtime; neps.get_in_progress_trial()`.

---

This module primarily handles the worker loop where important concepts are:
* **State**: The state of optimization is all of the configurations, their results and
 the current state of the optimizer.
* **Shared State**: Whenever a worker wishes to read or write any state, they will _lock_ the
 shared state, declaring themselves as operating on it. At this point, no other worker can
 access the shared state.
* **Optimizer Hydration**: This is the process through which an optimzier instance is _hydrated_
 with the Shared State so it can make a decision, i.e. for sampling. Equally we _serialize_
 the optimizer when writing it back to Shared State
* **Trial Lock**: When evaluating a configuration, a worker must _lock_ it to declared itself
 as evaluating it. This communicates to other workers that this configuration is pending

### Loop
We mark lines with `+` as the worker having locked the Shared State and `~` as the worker
having locked the Trial. The trial lock `~` is allowed to fail, in which case all steps
with a `~` are skipped and the loop continues.

1. + Check exit conditions
2. + Hydrate the optimizer
3. + Sample a new Trial
3. Unlock the Shared State
4. ~ Obtain a Trial Lock
5. ~ Set the global trial for this work to the current trial
6. ~ Evaluate the trial
7. ~+ Lock the shared state
8. ~+ Write the results of the config to disk
9. ~+ Update the optimizer if required (used budget for evaluating trial)
10. ~ Unlock the shared state
11. Unlock Trial Lock
"""
from __future__ import annotations

import inspect
import logging
import shutil
import time
import warnings
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Callable, TYPE_CHECKING
from enum import Enum

from neps.utils._locker import Locker
from neps.utils.files import empty_file, serialize, deserialize
from neps.types import ConfigLike, ERROR, ConfigResult, POST_EVAL_HOOK_SIGNATURE

if TYPE_CHECKING:
    from .optimizers.base_optimizer import BaseOptimizer

# Wait time between each successive poll to see if state can be grabbed
DEFAULT_STATE_POLL = 1
ENVIRON_STATE_POLL_KEY = "NEPS_STATE_POLL"

# Timeout before giving up on trying to grab the state, raising an error
DEFAULT_STATE_TIMEOUT = None
ENVIRON_STATE_TIMEOUT_KEY = "NEPS_STATE_TIMEOUT"


# TODO(eddiebergman): We should not do this...
warnings.simplefilter("always", DeprecationWarning)


# NOTE: As each NEPS process is only ever evaluating a single trial,
# this global can be retrieved in NePS and refers to what this process
# is currently evaluating.
_CURRENTLY_RUNNING_TRIAL_IN_PROCESS: Trial | None = None


def get_in_progress_trial() -> Trial | None:
    return _CURRENTLY_RUNNING_TRIAL_IN_PROCESS


def _set_in_progress_trial(trial: Trial | None) -> None:
    global _CURRENTLY_RUNNING_TRIAL_IN_PROCESS
    _CURRENTLY_RUNNING_TRIAL_IN_PROCESS = trial


@dataclass
class Trial:
    """A trial is a configuration and it's associated data.

    The object is considered mutable and the global trial currently being
    evaluated can be access using `get_in_progress_trial()`.

    Attributes:
        id: Unique identifier for the configuration
        config: The configuration to evaluate
        pipeline_dir: Directory where the configuration is evaluated
        prev_config_id: The id of the previous configuration evaluated for this trial.
        metadata: Additional metadata about the configuration
        results: The results of the evaluation, if any
        disk: The disk information of this trial such as paths and locks
    """

    id: str
    config: ConfigLike
    pipeline_dir: Path
    prev_config_id: str | None
    metadata: dict[str, Any]
    results: dict[str, Any] | ERROR | None = None

    disk: Trial.Disk = field(init=False)

    def __post_init__(self):
        if self.prev_config_id is not None:
            self.metadata["previous_config_id"] = self.prev_config_id
        self.disk = Trial.Disk(pipeline_dir=self.pipeline_dir)
        self.disk.pipeline_dir.mkdir(exist_ok=True, parents=True)

    @property
    def state(self) -> Trial.State:
        return self.disk.state

    def write_to_disk(self) -> Trial.Disk:
        serialize(self.config, self.disk.config_file)
        serialize(self.metadata, self.disk.metadata_file)

        if self.prev_config_id is not None:
            self.disk.previous_config_id_file.write_text(self.prev_config_id)

        if self.results is not None:
            serialize(self.results, self.disk.result_file)

        return self.disk

    class State(str, Enum):
        COMPLETE = "evaluated"
        IN_PROGRESS = "in_progress"
        PENDING = "pending"
        CORRUPTED = "corrupted"

        def __str__(self):
            return self.value

    @dataclass
    class Disk:
        pipeline_dir: Path

        id: str = field(init=False)
        config_file: Path = field(init=False)
        result_file: Path = field(init=False)
        metadata_file: Path = field(init=False)
        optimization_dir: Path = field(init=False)
        previous_config_id_file: Path = field(init=False)
        previous_pipeline_dir: Path | None = field(init=False)
        lock: Locker = field(init=False)

        def __post_init__(self):
            self.id = self.pipeline_dir.name[len("config_") :]
            self.config_file = self.pipeline_dir / "config.yaml"
            self.result_file = self.pipeline_dir / "result.yaml"
            self.metadata_file = self.pipeline_dir / "metadata.yaml"

            # NOTE: This is a bit of an assumption!
            self.optimization_dir = self.pipeline_dir.parent

            self.previous_config_id_file = self.pipeline_dir / "previous_config.id"
            if not empty_file(self.previous_config_id_file):
                with open(self.previous_config_id_file, "r") as f:
                    self.previous_config_id = f.read().strip()

                self.previous_pipeline_dir = (
                    self.pipeline_dir.parent / f"config_{self.previous_config_id}"
                )
            else:
                self.previous_pipeline_dir = None
            self.pipeline_dir.mkdir(exist_ok=True, parents=True)
            self.lock = Locker(self.pipeline_dir / ".config_lock")

        def config(self) -> ConfigLike:
            return deserialize(self.config_file)

        @property
        def state(self) -> Trial.State:
            if not empty_file(self.result_file):
                return Trial.State.COMPLETE
            elif self.lock.is_locked():
                return Trial.State.IN_PROGRESS
            elif not empty_file(self.config_file):
                return Trial.State.PENDING
            else:
                return Trial.State.CORRUPTED

        @classmethod
        def from_dir(cls, pipeline_dir: Path) -> Trial.Disk:
            return cls(pipeline_dir=pipeline_dir)

        def load(self) -> Trial:
            config = deserialize(self.config_file)
            if not empty_file(self.metadata_file):
                metadata = deserialize(self.metadata_file)
            else:
                metadata = {}

            if not empty_file(self.result_file):
                result = deserialize(self.result_file)
            else:
                result = None

            if not empty_file(self.previous_config_id_file):
                previous_config_id = self.previous_config_id_file.read_text().strip()
            else:
                previous_config_id = None

            return Trial(
                id=self.id,
                config=config,
                pipeline_dir=self.pipeline_dir,
                metadata=metadata,
                prev_config_id=previous_config_id,
                results=result,
            )

        # TODO(eddiebergman): Backwards compatibility on things that require the `ConfigResult`
        # Ideally, we just use `Trial` objects directly for things that need all informations
        # about trials.
        def to_result(
            self,
            config_transform: Callable[[ConfigLike], ConfigLike] | None = None,
        ) -> ConfigResult:
            config = deserialize(self.config_file)
            result = deserialize(self.result_file)
            metadata = deserialize(self.metadata_file)
            if config_transform is not None:
                config = config_transform(config)

            return ConfigResult(
                id=self.id,
                config=config,
                result=result,
                metadata=metadata,
            )


@dataclass
class SharedState:
    base_dir: Path
    create_dirs: bool = False

    lock: Locker = field(init=False)
    optimizer_state_file: Path = field(init=False)
    optimizer_info_file: Path = field(init=False)
    results_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.create_dirs:
            self.base_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.base_dir / "results"

        if self.create_dirs:
            self.results_dir.mkdir(exist_ok=True)

        self.lock = Locker(self.base_dir / ".decision_lock")
        self.optimizer_state_file = self.base_dir / ".optimizer_state.yaml"
        self.optimizer_info_file = self.base_dir / ".optimizer_info.yaml"

    def trial_refs(self) -> dict[Trial.State, list[Trial.Disk]]:
        refs = [
            Trial.Disk.from_dir(pipeline_dir=pipeline_dir)
            for pipeline_dir in self.results_dir.iterdir()
            if pipeline_dir.is_dir()
        ]
        by_state: dict[Trial.State, list[Trial.Disk]] = defaultdict(list)
        for ref in refs:
            by_state[ref.state].append(ref)

        return by_state

    def check_optimizer_info_on_disk_matches(
        self,
        optimizer_info: dict[str, Any],
        *,
        excluded_keys: Iterable[str] = ("searcher_name",),
    ) -> None:
        optimizer_info_copy = optimizer_info.copy()
        loaded_info = deserialize(self.optimizer_info_file)

        for key in excluded_keys:
            optimizer_info_copy.pop(key, None)
            loaded_info.pop(key, None)

        if optimizer_info_copy != loaded_info:
            raise ValueError(
                f"The sampler_info in the file {self.optimizer_info_file} is not valid. "
                f"Expected: {optimizer_info_copy}, Found: {loaded_info}"
            )


def _evaluate_config(
    trial: Trial,
    evaluation_fn: Callable[..., float | Mapping[str, Any]],
    logger: logging.Logger,
) -> tuple[ERROR | dict, float]:
    config = trial.config
    config_id = trial.id
    pipeline_directory = trial.disk.pipeline_dir
    previous_pipeline_directory = trial.disk.previous_pipeline_dir

    logger.info(f"Start evaluating config {config_id}")

    config = deepcopy(config)

    # If pipeline_directory and previous_pipeline_directory are included in the
    # signature we supply their values, otherwise we simply do nothing.
    directory_params = []

    evaluation_fn_params = inspect.signature(evaluation_fn).parameters
    if "pipeline_directory" in evaluation_fn_params:
        directory_params.append(pipeline_directory)
    if "previous_pipeline_directory" in evaluation_fn_params:
        directory_params.append(previous_pipeline_directory)

    try:
        result = evaluation_fn(*directory_params, **config)
    except Exception as e:
        logger.error(f"An error occured evaluating config '{config_id}': {config}.")
        logger.exception(e)
        result = "error"
    else:
        # Ensuring the result have the correct format that can be exploited by other functions
        if isinstance(result, Mapping):
            result = dict(result)
            if "loss" not in result:
                raise KeyError("The 'loss' should be provided in the evaluation result")
            loss = result["loss"]
        else:
            loss = result
            result = {}

        try:
            result["loss"] = float(loss)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "The evaluation result should be a dictionnary or a float but got"
                f" a `{type(loss)}` with value of {loss}"
            ) from e

    time_end = time.time()
    return result, time_end


def _try_remove_corrupted_configs(
    refs: Iterable[Trial.Disk],
    logger: logging.Logger,
) -> None:
    # If there are corrupted configs, we should remove them with a warning
    for ref in refs:
        logger.warning(f"Removing corrupted config {ref.id}")
        try:
            shutil.rmtree(ref.pipeline_dir)
        except Exception as e:
            logger.exception(e)


def _worker_should_continue(
    max_evaluations_total: int | None,
    continue_until_max_evaluation_completed: bool,
    refs: Mapping[Trial.State, list[Trial.Disk]],
    logger: logging.Logger,
) -> bool:
    # Check if we have reached the total amount of configurations to evaluated
    # (including pending evaluations possibly)
    if max_evaluations_total is None:
        return True

    logger.debug("Checking if max evaluations is reached")

    n_evaluated = len(refs[Trial.State.COMPLETE])
    n_inprogress = len(refs[Trial.State.IN_PROGRESS])

    n_counter = (
        n_evaluated
        if continue_until_max_evaluation_completed
        else n_evaluated + n_inprogress
    )
    return n_counter < max_evaluations_total


def launch_runtime(
    evaluation_fn: Callable[..., float | Mapping[str, Any]],
    sampler: BaseOptimizer,
    optimizer_info: dict,
    optimization_dir: Path | str,
    max_evaluations_total: int | None = None,
    max_evaluations_per_run: int | None = None,
    continue_until_max_evaluation_completed: bool = False,
    logger: logging.Logger | None = None,
    post_evaluation_hook: POST_EVAL_HOOK_SIGNATURE | None = None,
    overwrite_optimization_dir: bool = False,
    pre_load_hooks: Iterable[Callable[[BaseOptimizer], BaseOptimizer]] | None = None,
) -> None:
    # NOTE(eddiebergman): This was deprecated a while ago and called at
    # evaluate, now we just crash immediatly instead. Should probably
    # promote this check closer to the user, i.e. `neps.run()`
    evaluation_fn_params = inspect.signature(evaluation_fn).parameters
    if "previous_working_directory" in evaluation_fn_params:
        raise RuntimeError(
            "the argument: 'previous_working_directory' was deprecated. "
            f"In the function: '{evaluation_fn.__name__}', please,  "
            "use 'previous_pipeline_directory' instead. "
        )
    if "working_directory" in evaluation_fn_params:
        raise RuntimeError(
            "the argument: 'working_directory' was deprecated. "
            f"In the function: '{evaluation_fn.__name__}', please,  "
            "use 'pipeline_directory' instead. "
        )

    if logger is None:
        logger = logging.getLogger("neps")

    optimization_dir = Path(optimization_dir)

    # TODO(eddiebergman): Not sure how overwriting works with multiple workers....
    if overwrite_optimization_dir and optimization_dir.exists():
        logger.warning("Overwriting working_directory")
        shutil.rmtree(optimization_dir)

    shared_state = SharedState(optimization_dir, create_dirs=True)

    _poll = float(os.environ.get(ENVIRON_STATE_POLL_KEY, DEFAULT_STATE_POLL))
    _timeout = os.environ.get(ENVIRON_STATE_TIMEOUT_KEY, DEFAULT_STATE_TIMEOUT)
    if _timeout is not None:
        _timeout = float(_timeout)

    with shared_state.lock(poll=_poll, timeout=_timeout):
        if not shared_state.optimizer_info_file.exists():
            serialize(optimizer_info, shared_state.optimizer_info_file, sort_keys=False)
        else:
            shared_state.check_optimizer_info_on_disk_matches(optimizer_info)

    evaluations_in_this_run = 0
    while True:
        if (
            max_evaluations_per_run is not None
            and evaluations_in_this_run >= max_evaluations_per_run
        ):
            logger.info("Maximum evaluations per run is reached, shutting down")
            break

        with shared_state.lock(poll=_poll, timeout=_timeout):
            refs = shared_state.trial_refs()

            _try_remove_corrupted_configs(refs[Trial.State.CORRUPTED], logger)

            if not _worker_should_continue(
                max_evaluations_total,
                continue_until_max_evaluation_completed,
                refs,
                logger,
            ):
                logger.info("Maximum total evaluations is reached, shutting down")
                break

            # TODO(eddiebergman): I assume we should skip sampling and just go evaluate
            # pending configs?
            if any(refs[Trial.State.PENDING]):
                logger.warning(
                    f"There are {len(refs[Trial.State.PENDING])} configs that"
                    " were sampled, but have no worker assigned. Sometimes this is due to"
                    " a delay in the filesystem communication, but most likely some configs"
                    " crashed during their execution or a jobtime-limit was reached."
                )

            # While we have the decision lock, we will now sample with the optimizer in this process
            with sampler.using_state(shared_state.optimizer_state_file):
                if sampler.budget is not None and sampler.used_budget >= sampler.budget:
                    logger.info("Maximum budget reached, shutting down")
                    break

                logger.debug("Sampling a new configuration")
                if pre_load_hooks is not None:
                    for hook in pre_load_hooks:
                        sampler = hook(sampler)

                sampler.load_results(
                    previous_results={
                        ref.id: ref.to_result(config_transform=sampler.load_config)
                        for ref in refs[Trial.State.COMPLETE]
                    },
                    pending_evaluations={
                        ref.id: sampler.load_config(ref.config())
                        for ref in refs[Trial.State.IN_PROGRESS]
                    },
                )

                # TODO(eddiebergman): If we have some unified `Trial` like object,
                # we can just have them return this instead.
                config, config_id, prev_config_id = sampler.get_config_and_ids()

            trial = Trial(
                id=config_id,
                config=config,
                pipeline_dir=shared_state.results_dir / f"config_{config_id}",
                prev_config_id=prev_config_id,
                metadata={"time_sampled": time.time()},
            )
            trial.write_to_disk()
            logger.debug(f"Sampled config {config_id}")

            # Inform the global state of this process that we are evaluating this trial
            _set_in_progress_trial(trial)

        # Obtain the lock on this trial and evaluate it,
        # otherwise continue back to waiting to sampling
        with trial.disk.lock.try_lock() as acquired:
            if not acquired:
                continue

            # NOTE: Bit of an extra safety check but check that the trial is not complete
            if trial.disk.state == Trial.State.COMPLETE:
                continue

            result, time_end = _evaluate_config(trial, evaluation_fn, logger)
            meta: dict[str, Any] = {"time_end": time_end}

            # If this is set, it means we update the optimzier with the used
            # budget once we write the trial to disk and mark it as complete
            account_for_cost: bool = False
            eval_cost: float | None = None

            if result == "error":
                # TODO(eddiebergman): We should probably do something here...
                pass
            elif "cost" not in result and sampler.budget is not None:
                raise ValueError(
                    "The evaluation function result should contain "
                    f"a 'cost' field when used with a budget. Got {result}"
                )
            elif "cost" in result:
                eval_cost = float(result["cost"])
                account_for_cost = result.get("account_for_cost", True)
                meta["budget"] = {
                    "max": sampler.budget,
                    "used": sampler.used_budget,
                    "eval_cost": eval_cost,
                    "account_for_cost": account_for_cost,
                }

            trial.results = result
            trial.metadata.update(meta)

            with shared_state.lock(poll=_poll, timeout=_timeout):
                trial.write_to_disk()
                if account_for_cost:
                    assert eval_cost is not None
                    with sampler.using_state(shared_state.optimizer_state_file):
                        sampler.used_budget += eval_cost

            # 3. Anything the user might want to do after the evaluation
            if post_evaluation_hook is not None:
                post_evaluation_hook(
                    trial.config,
                    trial.id,
                    trial.pipeline_dir,
                    trial.results,
                    logger,
                )

            logger.info(f"Finished evaluating config {config_id}")

            evaluations_in_this_run += 1
