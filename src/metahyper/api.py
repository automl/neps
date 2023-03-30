from __future__ import annotations

import inspect
import logging
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping

from typing_extensions import Literal

from .config import Config
from .result import Result
from .sampler import Sampler
from .serialization import Serializer
from .states import DiskState

warnings.simplefilter("always", DeprecationWarning)


def _evaluate_config(
    config: Config,
    deserialized_config: Mapping[str, Any],
    evaluation_fn: Callable,
    logger: logging.Logger,
) -> tuple[Literal["error"] | dict[str, Any], dict[str, Any]]:
    deserialized_config = deepcopy(deserialized_config)
    logger.info(f"Start evaluating config {config.id}")

    # If pipeline_directory and previous_pipeline_directory are included in the
    # signature we supply their values, otherwise we simply do nothing.
    evaluation_fn_params = inspect.signature(evaluation_fn).parameters
    directory_params = []

    if "pipeline_directory" in evaluation_fn_params:
        directory_params.append(config.path)

    if "previous_pipeline_directory" in evaluation_fn_params:
        directory_params.append(config.previous_pipeline_dir)

    try:
        result = evaluation_fn(*directory_params, **deserialized_config)
    except Exception:
        logger.exception(
            f"An error occured during evaluation of config {config.id}: {deserialized_config}."
        )
        result = "error"
    finally:
        time_end = time.time()

    if result == "error":
        return result, {"time_end": time_end}

    values = {}

    # Ensuring the result have the correct format that can be exploited by other functions
    if isinstance(result, dict):
        if "loss" not in result:
            raise ValueError(f"The loss should be provided but got {result=}")

        loss = result["loss"]
    else:
        loss = result

    try:
        loss = float(loss)
    except (TypeError, ValueError) as e:
        raise type(e)("The result should be a dictionary or a float") from e

    values = {**result, "loss": loss}

    return values, {"time_end": time_end}


def remove_broken_configs(
    configs: list[Config],
    logger: logging.Logger,
    expected_suffix: str,
) -> None:
    logger.debug("Checking for broken configurations")

    for config in configs:
        existing_config_file = config.existing_config_file()

        if existing_config_file:
            existing_format = existing_config_file.suffix
            logger.warning(
                f"Found directory {config.path} with file {existing_config_file.name}. "
                f"But function was called with the serializer for "
                f"'{expected_suffix}' files, not '{existing_format}'."
            )
        else:
            logger.warning(
                f"Removing {config.path} as worker likely" " died during config sampling."
            )
            try:
                config.remove()
            except Exception as e:  # The worker doesn't need to crash for this
                logger.exception(f"Can't delete {config.path}: {e}")


def run(
    evaluation_fn: Callable,
    sampler: Sampler,
    optimization_dir: str | Path,
    max_evaluations_total: int | None = None,
    max_evaluations_per_run: int | None = None,
    continue_until_max_evaluation_completed: bool = False,
    development_stage_id: Hashable | None = None,
    task_id: Hashable | None = None,
    logger: logging.Logger | None = None,
    post_evaluation_hook: (
        Callable[[Mapping[str, Any], str, Path, Result, Any], Any] | None
    ) = None,
    overwrite_optimization_dir: bool = False,
) -> None:
    serializer = Serializer.default()
    logger = logger or logging.getLogger("metahyper")
    optimization_dir = Path(optimization_dir)

    # NOTE: It seems `development_stage_id` overwrites `task_id`. Perhaps there
    # should be a ValueError if both are specified.
    if task_id is not None:
        optimization_dir = optimization_dir / f"task_{task_id}"

    if development_stage_id is not None:
        optimization_dir = optimization_dir / f"dev_{development_stage_id}"

    state = DiskState(optimization_dir, logger=logger, clean=overwrite_optimization_dir)
    n_evals_this_run = 0

    while True:
        if (
            max_evaluations_per_run is not None
            and n_evals_this_run >= max_evaluations_per_run
        ):
            logger.info("Maximum evaluations per run is reached, shutting down")
            break

        # We start by locking up the global state, getting all the configurations
        with state.lock(timeout=1) as acquired:
            if not acquired:
                logger.debug("Waiting for lock")
                continue

            config_states = state.read(serializer)

            # Pylint seems confused but mypy isn't, just spelling it out for pylint
            config_states: dict[Config.Status, list[Config]]  # type: ignore

            logger.debug("Checking if max evaluations is reached")

            # Ensure that we don't evaluate more than the max evaluations
            eval_count = len(config_states[Config.Status.COMPLETE])
            if not continue_until_max_evaluation_completed:
                eval_count += len(config_states[Config.Status.ACTIVE])

            if max_evaluations_total is not None and eval_count >= max_evaluations_total:
                logger.info("Maximum total evaluations is reached, shutting down")
                break

            # Remove any borken configurations
            remove_broken_configs(
                configs=config_states[Config.Status.BROKEN]
                + config_states[Config.Status.UNKNOWN],
                logger=logger,
                expected_suffix=serializer.SUFFIX,
            )
            deserializer = sampler.parse_config

            results = {
                config.id: config.as_result(config_deserializer=deserializer)
                for config in config_states[Config.Status.COMPLETE]
            }

            pending = (
                config_states[Config.Status.ACTIVE] + config_states[Config.Status.FREE]
            )
            pending_configs = {
                config.id: deserializer(raw_config)
                for config in pending
                if (raw_config := config.raw_config()) is not None
            }

            with sampler.using_state(state, serializer):
                if sampler.exceeded_budget():
                    logger.info("Maximum budget reached, shutting down")
                    break

                config, deserialized_config = sampler._generate_config(  # pylint: disable=protected-access
                    base_result_directory=state.results_dir,
                    serializer=serializer,
                    logger=logger,
                    results=results,
                    pending_configs=pending_configs,
                )

            with config.lock(timeout=1) as acquired:
                if not acquired:
                    logger.debug("Waiting for config lock")
                    continue

                # 1. First, we evaluate the config
                result, metadata = _evaluate_config(
                    config, deserialized_config, evaluation_fn, logger
                )

                # 2. Then, we now dump all information to disk:
                serializer.dump(result, config.result_file)

                if result != "error":
                    if sampler.budget is not None and "cost" not in result:
                        raise ValueError(
                            "The evaluation function result should contain "
                            "a 'cost' field when used with a budget"
                            f"\nGot {result=} with budget {sampler.budget=}"
                        )

                    # Updating the global budget
                    if "cost" in result:
                        eval_cost = float(result["cost"])
                        account_for_cost = result.get("account_for_cost", True)
                        if account_for_cost:
                            with state.lock():
                                with sampler.using_state(state, serializer):
                                    sampler.used_budget += eval_cost

                        metadata["budget"] = {
                            "max": sampler.budget,
                            "used": sampler.used_budget,
                            "eval_cost": eval_cost,
                            "account_for_cost": account_for_cost,
                        }

                loaded_metadata = config.metadata() or {}
                serializer.dump({**loaded_metadata, **metadata}, config.metadata_file)

                # 3. Anything the user might want to do after the evaluation
                if post_evaluation_hook is not None:
                    post_evaluation_hook(
                        deserialized_config, config.id, config.path, result, logger
                    )

                logger.info(f"Finished evaluating config {config.id}")
                n_evals_this_run += 1
