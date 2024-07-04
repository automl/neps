from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from typing_extensions import Self

from neps.state.optimizer import OptimizerState
from neps.state.seeds import SeedSnapshot
from neps.state.shared import Shared
from neps.state.trial import Trial

if TYPE_CHECKING:
    from neps.state.trial import TrialID


@dataclass
class NePSPaths:
    root: Path
    results: Path
    optimizer_state: Path
    seed_state: Path

    @classmethod
    def from_directory(cls, directory: Path) -> NePSPaths:
        """Create a NePSPaths object from a directory."""
        return cls(
            root=directory,
            results=directory / "results",
            optimizer_state=directory / ".optimizer_state",
            seed_state=directory / ".seed_state",
        )


class LockOptions(NamedTuple):
    poll: float = 0.1
    timeout: float | None = None


@dataclass(kw_only=True)
class NePSState:
    directory: Path
    paths: NePSPaths

    _trials: dict[TrialID, Shared[Trial]]
    _optimizer_state: Shared[OptimizerState] | None
    _seed_state: Shared[SeedSnapshot] | None

    # Locked state
    _lock_options: LockOptions | None = field(
        default_factory=lambda: LockOptions(poll=0.1, timeout=None)
    )

    @property
    def unlocked(self) -> NePSState:
        return NePSState(
            directory=self.directory,
            paths=self.paths,
            _trials=self._trials,
            _lock_options=None,
            _optimizer_state=self._optimizer_state,
            _seed_state=self._seed_state,
        )

    def get_seeds(self) -> SeedSnapshot:
        if self._lock_options is None:
            if self._seed_state is None:

    @classmethod
    def get(cls, directory: Path) -> Self:
        paths = NePSPaths.from_directory(directory)
        trials: dict[TrialID, Shared[Trial]] = {}
        for config_path in paths.results.iterdir():
            if config_path.is_dir() and "config_" in config_path.name:
                trial_id = config_path.name.split("_", maxsplit=1)[1]

                # NOTE: We load_unsafe here as there may be long held locks
                # on trials as they're being evaluated.
                trial = Shared.load_unsafe(config_path, kls=Trial)
                trials[trial_id] = trial

        optimizer_state = Shared.load(paths.optimizer_state, kls=OptimizerState)
        seed_state = Shared.load(paths.seed_state, kls=SeedSnapshot)
        return NePSState(
            directory=directory,
            trials=trials,
            _optimizer_state=optimizer_state,
            _seed_state=seed_state,
            paths=paths,
        )
