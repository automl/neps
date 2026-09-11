"""Generation of the artifacts written to the `summary` folder of a study."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neps.optimizers.optimizer import Artifact, ArtifactType
from neps.plot.generic_plots import plot_incumbent_trajectory, plot_pareto_front
from neps.space.neps_spaces.neps_space import NepsCompatConverter
from neps.state.neps_state import NePSState, _deserialize_optimizer_info
from neps.state.trial import Trial
from neps.status.status import (
    _build_incumbent_content,
    _build_optimal_set_content,
    _initiate_summary_csv,
    status,
)
from neps.utils.files import get_file_writer

if TYPE_CHECKING:
    from neps.state.neps_state import FileLocker

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Container for tracking cumulative resource usage."""

    evaluations: int = 0
    cost: float = 0.0
    fidelities: float = 0.0
    time: float = 0.0

    def __iadd__(self, other: ResourceUsage) -> ResourceUsage:
        """Allows syntax: usage += other_usage."""
        self.evaluations += other.evaluations
        self.cost += other.cost
        self.fidelities += other.fidelities
        self.time += other.time
        return self

    def to_trajectory_dict(self) -> dict[str, float | int]:
        """Converts usage to the dictionary keys expected by the trajectory file."""
        return {
            "cumulative_evaluations": self.evaluations,
            "cumulative_cost": self.cost,
            "cumulative_fidelities": self.fidelities,
            "cumulative_time": self.time,
        }


def fidelity_name_of(space: Any) -> str | None:
    """The name a fidelity takes in a trial's config, if `space` has one."""
    if space is None:
        return None
    if getattr(space, "fidelity_attrs", None):
        name = next(iter(space.fidelity_attrs.keys()))
        return f"{NepsCompatConverter._ENVIRONMENT_PREFIX}{name}"
    if getattr(space, "fidelities", None):
        return str(next(iter(space.fidelities.keys())))
    return None


def resolve_fidelity_name(optimizer: Any, space: Any = None) -> str | None:
    """Find the fidelity name from an optimizer, falling back to `space`."""
    candidates = (
        getattr(optimizer, "space", None),
        getattr(optimizer, "pipeline_space", None),
        getattr(optimizer, "_pipeline", None),
        space,
    )
    for candidate in candidates:
        name = fidelity_name_of(candidate)
        if name is not None:
            return name
    return None


def calculate_total_resource_usage(
    trials: Mapping[str, Trial],
    fidelity_name: str | None = None,
    subset_worker_id: str | None = None,
    *,
    include_in_progress: bool = False,
) -> ResourceUsage:
    """Calculates total resources returning a typed usage object.

    Args:
        trials: Dictionary of trials to calculate from.
        fidelity_name: The key a fidelity takes in a trial's config, if any.
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

        if (
            fidelity_name
            and fidelity_name in trial.config
            and trial.config[fidelity_name] is not None
        ):
            usage.fidelities += trial.config[fidelity_name]

    return usage


def cumulative_resource_usage(
    trials: Sequence[Trial],
    fidelity_name: str | None = None,
) -> list[ResourceUsage]:
    """Calculates the running resource usage over `trials`, in the given order."""
    running = ResourceUsage()
    cumulative = []
    for trial in trials:
        running += calculate_total_resource_usage({trial.id: trial}, fidelity_name)
        cumulative.append(ResourceUsage(**asdict(running)))
    return cumulative


def _to_sequence(score: float | Sequence[float]) -> list[float]:
    if isinstance(score, Sequence):
        return [float(x) for x in score]
    return [float(score)]


def _is_dominated(candidate: float | Sequence[float], frontier: list[Trial]) -> bool:
    cand_seq = _to_sequence(candidate)

    for t in frontier:
        if t.report is None:
            continue
        f_seq = _to_sequence(t.report.objective_to_minimize)
        if len(f_seq) != len(cand_seq):
            continue
        if all(fi <= ci for fi, ci in zip(f_seq, cand_seq, strict=False)) and any(
            fi < ci for fi, ci in zip(f_seq, cand_seq, strict=False)
        ):
            return True
    return False


def _prune_and_add_to_frontier(candidate: Trial, frontier: list[Trial]) -> list[Trial]:
    if candidate.report is None:
        return frontier

    cand_seq = _to_sequence(candidate.report.objective_to_minimize)
    new_frontier: list[Trial] = []
    for t in frontier:
        if t.report is None:
            continue
        f_seq = _to_sequence(t.report.objective_to_minimize)
        if (
            len(f_seq) == len(cand_seq)
            and all(ci <= fi for ci, fi in zip(cand_seq, f_seq, strict=False))
            and any(ci < fi for ci, fi in zip(cand_seq, f_seq, strict=False))
        ):
            continue
        new_frontier.append(t)
    new_frontier.append(candidate)
    return new_frontier


@dataclass
class SummaryWriter:
    """Builds and persists every artifact of the `summary` folder.

    `collect` is pure: it turns the evaluated trials into a list of `Artifact`s.
    `write` is the only place that touches the disk, under a single lock so that
    the CSVs, the text files and the plots of one update land together.
    """

    root_directory: Path
    """The root directory of the run."""

    fidelity_name: str | None = None
    """The key a fidelity takes in a trial's config, if the space has one."""

    live_plots: bool = False
    """Whether to also produce the plots and the optimizer's own artifacts."""

    optimizer: Any | None = None
    """The optimizer, if at hand, to ask for its own artifacts."""

    summary_dir: Path = field(init=False)
    _full_path: Path = field(init=False, repr=False)
    _short_path: Path = field(init=False, repr=False)
    _locker: FileLocker = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root_directory = Path(self.root_directory).absolute().resolve()
        full_path, short_path, locker = _initiate_summary_csv(self.root_directory)
        self.summary_dir = full_path.parent
        self._full_path = full_path
        self._short_path = short_path
        self._locker = locker

    @property
    def full_csv_path(self) -> Path:
        """The CSV with one row per trial."""
        return self._full_path

    @property
    def short_csv_path(self) -> Path:
        """The CSV summarizing the run as a whole."""
        return self._short_path

    @classmethod
    def from_directory(
        cls,
        root_directory: Path | str,
        *,
        live_plots: bool = False,
    ) -> SummaryWriter:
        """Build a writer for an existing run."""
        root_directory = Path(root_directory)
        fidelity_name = None
        try:
            info = _deserialize_optimizer_info(root_directory / "optimizer_info.yaml")
            fidelity_name = info.get("fidelity_name")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not read the fidelity name from the run: {e}")

        return cls(
            root_directory=root_directory,
            fidelity_name=fidelity_name,
            live_plots=live_plots,
        )

    @property
    def best_config_path(self) -> Path:
        """The text file with the best config(s) of the run."""
        return self.summary_dir / "best_config.txt"

    @property
    def trajectory_path(self) -> Path:
        """The text file with the incumbent trajectory of the run."""
        return self.summary_dir / "best_config_trajectory.txt"

    def touch(self) -> None:
        """Create the summary folder and its (empty) files."""
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        with self._locker.lock():
            for path in (
                self._full_path,
                self._short_path,
                self.best_config_path,
                self.trajectory_path,
            ):
                path.touch(exist_ok=True)

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
            state = NePSState.create_or_load(path=self.root_directory, load_only=True)
            trials = state._trial_repo.get_valid_evaluated_trials()

        self.write(self.collect(trials, final=final))

    def collect(
        self,
        trials: Mapping[str, Trial],
        *,
        final: bool = False,
    ) -> list[Artifact]:
        """Turn the evaluated trials into the artifacts of the summary folder."""
        artifacts = self._csv_artifacts()
        if not trials:
            return artifacts

        sorted_trials: list[Trial] = sorted(
            trials.values(),
            key=lambda t: (
                t.metadata.time_sampled if t.metadata.time_sampled else float("inf")
            ),
        )
        is_mo = any(
            isinstance(trial.report.objective_to_minimize, list)  # type: ignore[union-attr]
            for trial in sorted_trials
        )

        # Rebuild a non-dominated frontier from the trials in chronological order.
        incumbent: list[Trial] = []
        frontier: list[Trial] = []
        trajectory_confs: dict[str, dict[str, float | int]] = {}
        cumulative_usage = cumulative_resource_usage(sorted_trials, self.fidelity_name)

        for evaluated_trial, usage in zip(sorted_trials, cumulative_usage, strict=True):
            assert evaluated_trial.report is not None  # for mypy
            new_trial_obj = evaluated_trial.report.objective_to_minimize

            if not _is_dominated(new_trial_obj, frontier):
                frontier = _prune_and_add_to_frontier(evaluated_trial, frontier)
                if not is_mo:
                    incumbent.append(evaluated_trial)
                config_dict = {
                    "score": new_trial_obj,
                    "trial_id": evaluated_trial.id,
                    "config": evaluated_trial.config,
                }
                if evaluated_trial.report.cost is not None:
                    config_dict["cost"] = evaluated_trial.report.cost

                config_dict.update(usage.to_trajectory_dict())
                trajectory_confs[evaluated_trial.id] = config_dict

        optimal_configs = [trajectory_confs[trial.id] for trial in frontier]
        incumbent_configs = [trajectory_confs[trial.id] for trial in incumbent]

        artifacts += self._trajectory_artifacts(
            incumbent_configs=incumbent_configs,
            optimal_configs=optimal_configs,
            final_usage=cumulative_usage[-1] if final else None,
        )

        if self.live_plots:
            artifacts += self._plot_artifacts(
                sorted_trials,
                cumulative_usage,
                incumbent_ids={trial.id for trial in incumbent},
                pareto_ids={trial.id for trial in frontier},
            )
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
        """`full.csv` with every trial and `short.csv` with the run's summary."""
        try:
            full_df, short = status(self.root_directory, print_summary=False)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to build the summary CSVs: {e}")
            return []

        return [
            Artifact(
                self._full_path.stem, full_df, ArtifactType.CSV, metadata={"index": True}
            ),
            Artifact(
                self._short_path.stem,
                short.to_frame(),
                ArtifactType.CSV,
                metadata={"index": True},
            ),
        ]

    def _trajectory_artifacts(
        self,
        *,
        incumbent_configs: list[dict],
        optimal_configs: list[dict],
        final_usage: ResourceUsage | None = None,
    ) -> list[Artifact]:  # TODO: could be removed by enriching the incumbent csv.
        """The incumbent trajectory and the final best config(s), as text."""
        artifacts = []
        if incumbent_configs:
            artifacts.append(
                Artifact(
                    "best_config_trajectory",
                    _build_incumbent_content(incumbent_configs),
                    ArtifactType.TEXT,
                )
            )

        if optimal_configs:
            best_config_text = _build_optimal_set_content(optimal_configs)
            if final_usage:
                best_config_text += "\n" + "-" * 80
                best_config_text += "\nFinal cumulative metrics (Assuming completed run):"
                for metric, value in final_usage.to_trajectory_dict().items():
                    best_config_text += f"\n{metric}: {value}"
            artifacts.append(Artifact("best_config", best_config_text, ArtifactType.TEXT))

        return artifacts

    def _plot_artifacts(
        self,
        trials: Sequence[Trial],
        cumulative_usage: Sequence[ResourceUsage],
        incumbent_ids: set[str],
        pareto_ids: set[str],
    ) -> list[Artifact]:
        assert trials[0].report is not None  # for mypy
        n_objectives = len(_to_sequence(trials[0].report.objective_to_minimize))  # type: ignore[arg-type]
        try:
            if n_objectives == 1:
                return plot_incumbent_trajectory(trials, cumulative_usage, incumbent_ids)
            if n_objectives == 2:
                return plot_pareto_front(trials, pareto_ids)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to create the summary plot: {e}")
            return []

        logger.debug("No summary plot for %d objectives.", n_objectives)
        return []

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
