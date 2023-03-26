from __future__ import annotations

import inspect
import logging
import shutil
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._locker import Locker
from .utils import YamlSerializer, find_files, non_empty_file

warnings.simplefilter("always", DeprecationWarning)


@dataclass
class ConfigResult:
    config: Any
    result: dict
    metadata: dict

    def __lt__(self, other):
        return self.result["loss"] < other.result["loss"]


class Sampler(ABC):
    # pylint: disable=no-self-use,unused-argument
    def __init__(self, budget: int | float | None = None):
        self.used_budget: int | float = 0
        self.budget = budget

    def get_state(self) -> Any:
        """Return a state for the sampler that will be used in every other thread"""
        state = {
            "used_budget": self.used_budget,
        }
        if self.budget is not None:
            state["remaining_budget"] = self.budget - self.used_budget
        return state

    def load_state(self, state: dict[str, Any]):
        """Load a state for the sampler shared accross threads"""
        self.used_budget = state["used_budget"]

    @contextmanager
    def using_state(self, state_file: Path, serializer):
        if state_file.exists():
            self.load_state(serializer.load(state_file))
        yield self

        serializer.dump(self.get_state(), state_file)

    def load_results(
        self, results: dict[Any, ConfigResult], pending_configs: dict[Any, ConfigResult]
    ) -> None:
        return

    @abstractmethod
    def get_config_and_ids(self) -> tuple[Any, str, str | None]:
        """Sample a new configuration

        Returns:
            config: serializable object representing the configuration
            config_id: unique identifier for the configuration
            previous_config_id: if provided, id of a previous on which this
                configuration is based
        """
        raise NotImplementedError

    def load_config(self, config: Any):
        """Transform a serialized object into a configuration object"""
        return config


class Configuration:
    """If the configuration is not a simple dictionary, it should inherit from
    this object and define the 'hp_values' method"""

    def hp_values(self):
        """Should return a dictionary of the hyperparameter values"""
        raise NotImplementedError


def _load_sampled_paths(optimization_dir: Path | str, serializer, logger):
    optimization_dir = Path(optimization_dir)
    base_result_directory = optimization_dir / "results"
    logger.debug(f"Loading results from {base_result_directory}")

    previous_paths, pending_paths = {}, {}
    for config_dir in base_result_directory.iterdir():
        if not config_dir.is_dir():
            continue
        config_id = config_dir.name[len("config_") :]
        config_file = config_dir / f"config{serializer.SUFFIX}"
        result_file = config_dir / f"result{serializer.SUFFIX}"

        if non_empty_file(result_file):
            previous_paths[config_id] = (config_dir, config_file, result_file)
        elif non_empty_file(config_file):
            pending_paths[config_id] = (config_dir, config_file)
        else:
            existing_config = find_files(
                config_dir, ["config"], any_suffix=True, check_nonempty=True
            )
            if existing_config:
                existing_format = existing_config[0].suffix
                logger.warning(
                    f"Found directory {config_dir} with file {existing_config[0].name}. "
                    f"But function was called with the serializer for "
                    f"'{serializer.SUFFIX}' files, not '{existing_format}'."
                )
            else:
                # Should probably warn the user somehow about this, although it is not
                # dangerous
                logger.warning(
                    f"Removing {config_dir} as worker died during config sampling."
                )
                try:
                    shutil.rmtree(str(config_dir))
                except Exception:  # The worker doesn't need to crash for this
                    logger.exception(f"Can't delete {config_dir}")

    return previous_paths, pending_paths


def _read_config_result(result_dir: Path | str, serializer: YamlSerializer):
    result_dir = Path(result_dir)
    config = serializer.load_config(result_dir / "config")
    result = serializer.load(result_dir / "result")
    metadata = serializer.load(result_dir / "metadata")
    return ConfigResult(config, result, metadata)


def read(optimization_dir: Path | str, serializer=None, logger=None, do_lock=True):
    optimization_dir = Path(optimization_dir)

    if logger is None:
        logger = logging.getLogger("metahyper")

    if do_lock:
        decision_lock_file = optimization_dir / ".decision_lock"
        decision_lock_file.touch(exist_ok=True)
        decision_locker = Locker(decision_lock_file, logger.getChild("_locker"))
        while not decision_locker.acquire_lock():
            time.sleep(2)

    if serializer is None:
        serializer = YamlSerializer()

    previous_paths, pending_paths = _load_sampled_paths(
        optimization_dir, serializer, logger
    )
    previous_results, pending_configs, pending_configs_free = {}, {}, {}

    for config_id, (config_dir, _, _) in previous_paths.items():
        previous_results[config_id] = _read_config_result(config_dir, serializer)

    for config_id, (config_dir, config_file) in pending_paths.items():
        pending_configs[config_id] = serializer.load_config(config_file)

        config_lock_file = config_dir / ".config_lock"
        config_locker = Locker(config_lock_file, logger.getChild("_locker"))
        if config_locker.acquire_lock():
            pending_configs_free[config_id] = pending_configs[config_id]

    logger.debug(
        f"Read in {len(previous_results)} previous results and "
        f"{len(pending_configs)} pending evaluations "
        f"({len(pending_configs_free)} without a worker)"
    )
    logger.debug(
        f"Read in previous_results={previous_results}, "
        f"pending_configs={pending_configs}, "
        f"and pending_configs_free={pending_configs_free}, "
    )

    if do_lock:
        decision_locker.release_lock()
    return previous_results, pending_configs, pending_configs_free


def _check_max_evaluations(
    optimization_dir,
    max_evaluations,
    serializer,
    logger,
    continue_until_max_evaluation_completed,
):
    logger.debug("Checking if max evaluations is reached")

    previous_results, pending_configs, pending_configs_free = read(
        optimization_dir, serializer, logger
    )
    evaluation_count = len(previous_results)

    # Taking into account pending evaluations
    if not continue_until_max_evaluation_completed:
        evaluation_count += len(pending_configs) - len(pending_configs_free)

    return evaluation_count >= max_evaluations


def _sample_config(optimization_dir, sampler, serializer, logger):
    # First load the results and state of the optimizer
    previous_results, pending_configs, pending_configs_free = read(
        optimization_dir, serializer, logger, do_lock=False
    )

    base_result_directory = optimization_dir / "results"

    logger.debug("Sampling a new configuration")
    sampler.load_results(previous_results, pending_configs)
    config, config_id, previous_config_id = sampler.get_config_and_ids()

    pipeline_directory = base_result_directory / f"config_{config_id}"
    pipeline_directory.mkdir(exist_ok=True)

    if pending_configs_free:
        logger.warning(
            f"There are {len(pending_configs_free)} configs that were sampled, but have no worker assigned. Sometimes this is due to a delay in the filesystem communication, but most likely some configs crashed during their execution or a jobtime-limit was reached."
        )

    if previous_config_id is not None:
        previous_config_id_file = pipeline_directory / "previous_config.id"
        previous_config_id_file.write_text(previous_config_id)  # TODO: Get rid of this
        serializer.dump(
            {"time_sampled": time.time(), "previous_config_id": previous_config_id},
            pipeline_directory / "metadata",
        )
        previous_pipeline_directory = Path(
            base_result_directory, f"config_{previous_config_id}"
        )
    else:
        serializer.dump({"time_sampled": time.time()}, pipeline_directory / "metadata")
        previous_pipeline_directory = None

    # We want this to be the last action in sampling to catch potential crashes
    serializer.dump(config, pipeline_directory / "config")

    logger.debug(f"Sampled config {config_id}")
    return (
        config_id,
        config,
        pipeline_directory,
        previous_pipeline_directory,
    )


def _evaluate_config(
    config_id,
    config,
    pipeline_directory,
    evaluation_fn,
    previous_pipeline_directory,
    logger,
):
    if isinstance(config, Configuration):
        config = config.hp_values()
    config = deepcopy(config)
    logger.info(f"Start evaluating config {config_id}")
    try:
        # If pipeline_directory and previous_pipeline_directory are included in the
        # signature we supply their values, otherwise we simply do nothing.
        evaluation_fn_params = inspect.signature(evaluation_fn).parameters
        directory_params = []
        if "pipeline_directory" in evaluation_fn_params:
            directory_params.append(pipeline_directory)
        elif "working_directory" in evaluation_fn_params:
            warnings.warn(
                "the argument: 'working_directory' is deprecated. "
                "In the function: '{}', please, "
                "use 'pipeline_directory' instead. "
                "version==0.5.5".format(evaluation_fn.__name__),
                DeprecationWarning,
                stacklevel=2,
            )
            directory_params.append(pipeline_directory)

        if "previous_pipeline_directory" in evaluation_fn_params:
            directory_params.append(previous_pipeline_directory)
        elif "previous_working_directory" in evaluation_fn_params:
            warnings.warn(
                "the argument: 'previous_working_directory' is deprecated. "
                "In the function: '{}', please,  "
                "use 'previous_pipeline_directory' instead. "
                "version==0.5.5".format(evaluation_fn.__name__),
                DeprecationWarning,
                stacklevel=2,
            )
            directory_params.append(previous_pipeline_directory)

        result = evaluation_fn(
            *directory_params,
            **config,
        )

        # Ensuring the result have the correct format that can be exploited by other functions
        if isinstance(result, dict):
            try:
                result["loss"] = float(result["loss"])
            except KeyError as e:
                raise ValueError("The loss should value should be provided") from e
            except (TypeError, ValueError) as e:
                raise ValueError("The loss should be a float") from e
        else:
            try:
                result = float(result)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "The evaluation result should be a dictionnary or a float"
                ) from e
            result = {"loss": result}
    except Exception:
        logger.exception(
            f"An error occured during evaluation of config {config_id}: " f"{config}."
        )
        result = "error"

    return result, {"time_end": time.time()}


def run(
    evaluation_fn,
    sampler: Sampler,
    optimization_dir,
    max_evaluations_total=None,
    max_evaluations_per_run=None,
    continue_until_max_evaluation_completed=False,
    development_stage_id=None,
    task_id=None,
    logger=None,
    post_evaluation_hook=None,
    overwrite_optimization_dir=False,
):
    serializer = YamlSerializer(sampler.load_config)
    if logger is None:
        logger = logging.getLogger("metahyper")

    if task_id is not None:
        optimization_dir = Path(optimization_dir) / f"task_{task_id}"
    if development_stage_id is not None:
        optimization_dir = Path(optimization_dir) / f"dev_{development_stage_id}"

    optimization_dir = Path(optimization_dir)
    if overwrite_optimization_dir and optimization_dir.exists():
        logger.warning("Overwriting working_directory")
        shutil.rmtree(optimization_dir)

    sampler_state_file = optimization_dir / ".optimizer_state.yaml"
    base_result_directory = optimization_dir / "results"
    base_result_directory.mkdir(parents=True, exist_ok=True)

    decision_lock_file = optimization_dir / ".decision_lock"
    decision_lock_file.touch(exist_ok=True)
    decision_locker = Locker(decision_lock_file, logger.getChild("_locker"))

    evaluations_in_this_run = 0
    while True:
        if max_evaluations_total is not None and _check_max_evaluations(
            optimization_dir,
            max_evaluations_total,
            serializer,
            logger,
            continue_until_max_evaluation_completed,
        ):
            logger.info("Maximum total evaluations is reached, shutting down")
            break

        if (
            max_evaluations_per_run is not None
            and evaluations_in_this_run >= max_evaluations_per_run
        ):
            logger.info("Maximum evaluations per run is reached, shutting down")
            break

        if decision_locker.acquire_lock():
            try:
                with sampler.using_state(sampler_state_file, serializer):
                    if sampler.budget is not None:
                        if sampler.used_budget >= sampler.budget:
                            logger.info("Maximum budget reached, shutting down")
                            break
                    (
                        config_id,
                        config,
                        pipeline_directory,
                        previous_pipeline_directory,
                    ) = _sample_config(optimization_dir, sampler, serializer, logger)

                config_lock_file = pipeline_directory / ".config_lock"
                config_lock_file.touch(exist_ok=True)
                config_locker = Locker(config_lock_file, logger.getChild("_locker"))
                config_lock_acquired = config_locker.acquire_lock()
            finally:
                decision_locker.release_lock()

            if config_lock_acquired:
                try:
                    # 1. First, we evaluate the config
                    result, metadata = _evaluate_config(
                        config_id,
                        config,
                        pipeline_directory,
                        evaluation_fn,
                        previous_pipeline_directory,
                        logger,
                    )

                    # 2. Then, we now dump all information to disk:
                    serializer.dump(result, pipeline_directory / "result")

                    if result != "error":
                        # Updating the global budget
                        if "cost" in result:
                            eval_cost = float(result["cost"])
                            account_for_cost = result.get("account_for_cost", True)
                            if account_for_cost:
                                with decision_locker.acquire_force(time_step=1):
                                    with sampler.using_state(
                                        sampler_state_file, serializer
                                    ):
                                        sampler.used_budget += eval_cost

                            metadata["budget"] = {
                                "max": sampler.budget,
                                "used": sampler.used_budget,
                                "eval_cost": eval_cost,
                                "account_for_cost": account_for_cost,
                            }
                        elif sampler.budget is not None:
                            raise ValueError(
                                "The evaluation function result should contain "
                                "a 'cost' field when used with a budget"
                            )

                    config_metadata = serializer.load(pipeline_directory / "metadata")
                    config_metadata.update(metadata)
                    serializer.dump(config_metadata, pipeline_directory / "metadata")

                    # 3. Anything the user might want to do after the evaluation
                    if post_evaluation_hook is not None:
                        post_evaluation_hook(
                            config, config_id, pipeline_directory, result, logger
                        )
                    else:
                        logger.info(f"Finished evaluating config {config_id}")
                finally:
                    config_locker.release_lock()

                evaluations_in_this_run += 1
        else:
            time.sleep(3)
