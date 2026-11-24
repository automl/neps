"""Generation of the artifacts written to the `summary` folder of a study."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scalefm.optimizers.optimizer import Artifact, ArtifactType
from scalefm.plot.scaling_plots import plot_scaling
from scalefm.space.scalefm_spaces.parameters import PipelineSpace
from scalefm.space.scalefm_spaces.scalefm_space import (
    convert_scalefm_to_classic_search_space,
)
from scalefm.state.scalefm_state import ScaleFMState, _deserialize_optimizer_info
from scalefm.state.trial import Trial
from scalefm.status.status import _initiate_summary_csv, status
from scalefm.utils.files import get_file_writer

if TYPE_CHECKING:
    from scalefm.space import SearchSpace
    from scalefm.state.scalefm_state import FileLocker

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Container for tracking cumulative resource usage."""

    evaluations: int = 0
    cost: float = 0.0
    time: float = 0.0

    def __iadd__(self, other: ResourceUsage) -> ResourceUsage:
        """Allows syntax: usage += other_usage."""
        self.evaluations += other.evaluations
        self.cost += other.cost
        self.time += other.time
        return self


def calculate_total_resource_usage(
    trials: Mapping[str, Trial],
    subset_worker_id: str | None = None,
    *,
    include_in_progress: bool = False,
) -> ResourceUsage:
    """Calculates total resources returning a typed usage object.

    Args:
        trials: Dictionary of trials to calculate from.
        subset_worker_id: If provided, only calculates for
            trials evaluated by this worker ID.
        include_in_progress: Whether to include incomplete trials.
    """
    relevant_trials = list(trials.values())
    if subset_worker_id is not None:
        relevant_trials = [
            t
            for t in relevant_trials
            if t.metadata.evaluating_worker_id == subset_worker_id
        ]

    usage = ResourceUsage()

    for trial in relevant_trials:
        if not (
            trial.report is not None
            or (include_in_progress and trial.metadata.state == Trial.State.EVALUATING)
        ):
            continue
        usage.evaluations += 1
        if trial.report and trial.report.cost is not None:
            usage.cost += trial.report.cost

        # Handle time: either from report or calculate from metadata
        if trial.report and trial.report.evaluation_duration is not None:
            usage.time += trial.report.evaluation_duration
        elif (
            trial.metadata.time_started is not None
            and trial.metadata.time_end is not None
        ):
            usage.time += trial.metadata.time_end - trial.metadata.time_started

    return usage


@dataclass
class SummaryWriter:
    """Builds and persists every artifact of the `summary` folder.

    `collect` is pure: it turns the evaluated trials into a list of `Artifact`s.
    `write` is the only place that touches the disk, under a single lock so that
    the CSVs, the text files and the plots of one update land together.
    """

    root_directory: Path
    """The root directory of the run."""

    live_plots: bool = False
    """Whether to also produce the plots and the optimizer's own artifacts."""

    optimizer: Any | None = None
    """The optimizer, if at hand, to ask for its own artifacts."""

    summary_dir: Path = field(init=False)
    _full_path: Path = field(init=False, repr=False)
    _locker: FileLocker = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root_directory = Path(self.root_directory).absolute().resolve()
        full_path, locker = _initiate_summary_csv(self.root_directory)
        self.summary_dir = full_path.parent
        self._full_path = full_path
        self._locker = locker

    @property
    def full_csv_path(self) -> Path:
        """The CSV with one row per trial."""
        return self._full_path

    @classmethod
    def from_directory(
        cls,
        root_directory: Path | str,
        *,
        live_plots: bool = False,
    ) -> SummaryWriter:
        """Build a writer for an existing run."""
        return cls(root_directory=Path(root_directory), live_plots=live_plots)

    def touch(self) -> None:
        """Create the summary folder and its (empty) files."""
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        with self._locker.lock():
            self._full_path.touch(exist_ok=True)

    def update(
        self,
        trials: Mapping[str, Trial] | None = None,
        *,
        final: bool = False,
    ) -> None:
        """Rebuild every summary artifact and write it out.

        Args:
            trials: The evaluated trials with a valid report. Loaded from disk
                if not given.
            final: Whether this is the last update of the run
        """
        if trials is None:
            state = ScaleFMState.create_or_load(path=self.root_directory, load_only=True)
            trials = state._trial_repo.get_valid_evaluated_trials(refresh_cache=False)

        artifacts = self.collect(trials, final=final)
        self.write(artifacts)

    def collect(
        self,
        trials: Mapping[str, Trial],
        *,
        final: bool = False,  # noqa: ARG002
    ) -> list[Artifact]:
        """Turn the evaluated trials into the artifacts of the summary folder."""
        artifacts = self._csv_artifacts()
        if not trials or not self.live_plots:
            return artifacts

        sorted_trials: list[Trial] = sorted(
            trials.values(),
            key=lambda t: (
                t.metadata.time_sampled if t.metadata.time_sampled else float("inf")
            ),
        )
        artifacts += self._plot_artifacts(sorted_trials)
        artifacts += self._optimizer_artifacts(trials)
        return artifacts

    def write(self, artifacts: Sequence[Artifact]) -> None:
        """Write out artifacts, under a single lock."""
        if not artifacts:
            return

        self.summary_dir.mkdir(parents=True, exist_ok=True)
        with self._locker.lock():
            for artifact in artifacts:
                try:
                    writer = get_file_writer(artifact.artifact_type.value)
                    file_path = self.summary_dir / artifact.name

                    accepted = set(inspect.signature(writer.write).parameters)
                    unknown = set(artifact.metadata) - accepted
                    if unknown:
                        raise TypeError(
                            f"metadata key(s) {sorted(unknown)} are not accepted by "
                            f"{type(writer).__name__}.write(); valid keys:"
                            f" {sorted(accepted)}"
                        )

                    writer.write(artifact.content, file_path, **artifact.metadata)
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"Failed to save artifact '{artifact.name}' "
                        f"(type={artifact.artifact_type.value}): {e}"
                    )
                    continue

    def _csv_artifacts(self) -> list[Artifact]:
        """`full.csv` with every trial."""
        try:
            full_df, _short = status(self.root_directory, print_summary=False)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to build the summary CSV: {e}")
            return []

        return [
            Artifact(
                self._full_path.stem, full_df, ArtifactType.CSV, metadata={"index": True}
            ),
        ]

    def _plot_artifacts(self, trials: Sequence[Trial]) -> list[Artifact]:
        try:
            return plot_scaling(
                trials, self._run_fit(), self._target_costs(), self._search_space()
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to create the summary plot: {e}")
            return []

    def _run_fit(self) -> str:
        """The scaling law the run's optimizer names."""
        try:
            info = _deserialize_optimizer_info(
                self.root_directory / "optimizer_info.yaml"
            )
        except Exception:  # noqa: BLE001
            return "power_law"
        return info.get("derived", {}).get("predictor", "power_law")

    def _target_costs(self) -> dict[str, list[float]] | None:
        """The target costs of the last `scalefm.predict()`, if any."""
        path = self.summary_dir / "targets.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _search_space(self) -> SearchSpace | None:
        """The search space of the run, of the classic kind the scaling laws take."""
        try:
            state = ScaleFMState.create_or_load(path=self.root_directory, load_only=True)
            space = state.lock_and_get_search_space()
            if isinstance(space, PipelineSpace):
                return convert_scalefm_to_classic_search_space(space)
        except Exception:  # noqa: BLE001
            return None
        return space

    def _optimizer_artifacts(self, trials: Mapping[str, Trial]) -> list[Artifact]:
        """Whatever the optimizer itself wants persisted, if it offers any."""
        if self.optimizer is None or not hasattr(self.optimizer, "get_trial_artifacts"):
            return []
        try:
            artifacts = self.optimizer.get_trial_artifacts(trials=trials)
        except Exception as e:
            logger.error(f"Failed to collect optimizer artifacts: {e}", exc_info=True)
            return []
        return list(artifacts) if artifacts is not None else []
