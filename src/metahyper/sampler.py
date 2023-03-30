from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Iterator, Mapping, TypeVar

from typing_extensions import Self

from neps.search_spaces.search_space import SearchSpace

from .config import Config
from .serialization import Serializer

if TYPE_CHECKING:
    from .states import DiskState

DeserializedConfig = TypeVar("DeserializedConfig", bound=Mapping)


class Sampler(ABC, Generic[DeserializedConfig]):
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

    def load_state(self, state: dict[str, Any]) -> None:
        """Load a state for the sampler shared accross threads"""
        self.used_budget = state["used_budget"]

    def exceeded_budget(self) -> bool:
        """Return True if the budget is exceeded"""
        if self.budget is None:
            return False

        return self.used_budget >= self.budget

    @contextmanager
    def using_state(self, state: DiskState, serializer: Serializer) -> Iterator[Self]:
        with state.lock():
            sampler_state_file = state.sampler_state_file
            if sampler_state_file.exists():
                self.load_state(serializer.load(sampler_state_file))

            yield self

            serializer.dump(self.get_state(), sampler_state_file)

    def load_results(
        self,
        results: dict[str, Config.Result[DeserializedConfig]],
        pending_configs: dict[str, DeserializedConfig],
    ) -> None:
        return

    @abstractmethod
    def get_config_and_ids(self) -> tuple[DeserializedConfig, str, str | None]:
        """Sample a new configuration

        Returns:
            config: serializable object representing the configuration
            config_id: unique identifier for the configuration
            previous_config_id: if provided, id of a previous on which this
                configuration is based
        """
        raise NotImplementedError

    @abstractmethod
    def parse_config(self, disk_config: Mapping[str, Any]) -> DeserializedConfig:
        """Transform a serialized object into a configuration object"""
        ...

    def _generate_config(
        self,
        base_result_directory: Path,
        serializer: Serializer,
        logger: logging.Logger,
        results: dict[str, Config.Result[DeserializedConfig]],
        pending_configs: dict[str, DeserializedConfig],
    ) -> tuple[Config, DeserializedConfig]:
        # First load the results and state of the optimizer
        # previous_results, pending_configs, pending_configs_free = read(
        # optimization_dir, serializer, logger, do_lock=False
        # )
        logger.debug("Sampling a new configuration")

        self.load_results(results, pending_configs)

        deserialized_config, config_id, previous_config_id = self.get_config_and_ids()

        pipeline_directory = base_result_directory / f"config_{config_id}"
        pipeline_directory.mkdir(exist_ok=True)

        if pending_configs:
            logger.warning(
                f"There are {len(pending_configs)} configs that were sampled, but have no"
                " worker assigned. Sometimes this is due to a delay in the filesystem"
                " communication, but most likely some configs crashed during their"
                " execution or a jobtime-limit was reached."
            )

        if previous_config_id is not None:
            previous_config_id_file = pipeline_directory / "previous_config.id"
            previous_config_id_file.write_text(previous_config_id)

            previous_pipeline_directory = (
                base_result_directory / f"config_{previous_config_id}"
            )
            previous_config_dir_file = pipeline_directory / "previous_config.dir"
            previous_config_dir_file.write_text(str(previous_pipeline_directory))

            metadata = {
                "time_sampled": time.time(),
                "previous_config_id": previous_config_id,
            }
        else:
            metadata = {"time_sampled": time.time()}
            previous_pipeline_directory = None

        serializer.dump(metadata, pipeline_directory / "metadata")

        # We want this to be the last action in sampling to catch potential crashes
        serializer.dump(deserialized_config, pipeline_directory / "config")

        logger.debug(f"Sampled config {config_id}")
        config = Config(pipeline_directory, serializer=serializer)
        return config, deserialized_config
