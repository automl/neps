from __future__ import annotations

import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, overload

from typing_extensions import Literal

from ._locker import Locker
from .config import Config
from .serialization import Serializer


class DiskState:
    def __init__(
        self,
        optimization_dir: str | Path,
        clean: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("metahyper")
        self.optimization_dir = Path(optimization_dir)

        if clean and self.optimization_dir.exists():
            self.logger.warning(
                f"Working from a clean optimization state at {optimization_dir}"
            )
            shutil.rmtree(str(self.optimization_dir))

        self.optimization_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def results_dir(self) -> Path:
        return self.optimization_dir / "results"

    @property
    def sampler_state_file(self) -> Path:
        return self.optimization_dir / ".optimizer_state.yaml"

    # With no lock, we can read the results
    @overload
    def read(
        self,
        serializer: Serializer | None = ...,
        *,
        locked: Literal[False],
        timeout: float | None = None,
    ) -> dict[Config.Status, list[Config]]:
        ...

    # With a lock but not timeout, we eventually read the results
    @overload
    def read(
        self,
        serializer: Serializer | None = ...,
        *,
        locked: Literal[True] = True,
        timeout: None = None,
    ) -> dict[Config.Status, list[Config]]:
        ...

    # With a lock and timeout, we may not read the results
    @overload
    def read(
        self,
        serializer: Serializer | None = ...,
        *,
        locked: Literal[True],
        timeout: float,
    ) -> dict[Config.Status, list[Config]] | None:
        ...

    def read(
        self,
        serializer: Serializer | None = None,
        *,
        locked: bool = True,
        timeout: float | None = None,
    ) -> dict[Config.Status, list[Config]] | None:
        if serializer is None:
            serializer = Serializer.default()

        with self.lock(active=locked, timeout=timeout) as acquired:
            if not acquired:
                self.logger.debug(
                    f"Could not acquire lock on {self.lock_path} with `read`"
                )
                return None

            self.logger.debug(f"Reading results from {self.results_dir}")

            configs = [
                Config(path, serializer)
                for path in self.results_dir.iterdir()
                if path.is_dir()
            ]

            # Get all configs with their status
            configs_with_status: dict[Config.Status, list[Config]] = {
                state: [] for state in Config.Status
            }
            for config in configs:
                status = config.status()
                configs_with_status[status].append(config)

        n_total = sum(len(configs) for configs in configs_with_status.values())
        self.logger.debug(f"Read in {n_total} configs:\n{configs_with_status}")

        return configs_with_status

    @contextmanager
    def lock(
        self, active: bool = True, *, timeout: float | None = None
    ) -> Iterator[bool]:
        """Return True if the state is active, False otherwise"""
        if active:
            lock = Locker.local(self.lock_path)
            with lock.lock(timeout=timeout) as acquired:
                yield acquired
        else:
            yield True

    @property
    def lock_path(self) -> Path:
        return self.optimization_dir / ".decision.lock"
