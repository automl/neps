"""The tblogger module provides a simplified interface for logging to TensorBoard."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from torch.utils.tensorboard.writer import SummaryWriter

from neps.runtime import (
    get_in_progress_trial,
    get_workers_neps_state,
    is_in_progress_trial_set,
    register_notify_trial_end,
)
from neps.status.status import status
from neps.utils.common import get_initial_directory

if TYPE_CHECKING:
    from neps.state.trial import Trial

logger = logging.getLogger(__name__)


class tblogger:  # noqa: N801
    """Provides a simplified interface for logging NePS trials to TensorBoard."""

    config_working_directory: ClassVar[Path | None] = None
    optimizer_dir: ClassVar[Path | None] = None
    config_previous_directory: ClassVar[Path | None] = None

    write_incumbent: ClassVar[bool | None] = None

    config_writer: ClassVar[SummaryWriter | None] = None
    summary_writer: ClassVar[SummaryWriter | None] = None

    @classmethod
    def initiate_internal_configurations(
        cls,
        root_directory: Path | None = None,
        pipeline_directory: Path | None = None,
        previous_pipeline_directory: Path | None = None,
    ) -> None:
        """Initialize internal directories and configuration for TensorBoard logging.

        This function determines the working and previous trial directories either
        from an in-progress trial or from provided arguments.

        Args:
            root_directory (Path | str | None): The root optimization directory.
            pipeline_directory (Path | str | None): Current trial directory.
            previous_pipeline_directory (Path | str | None): Previous trial directory.
        """
        if not is_in_progress_trial_set() and not (root_directory and pipeline_directory):
            raise RuntimeError(
                "Cannot determine directories for TensorBoard logging. "
                "Provide `root_directory`, `pipeline_directory`, and optionally "
                "`previous_pipeline_directory`."
            )

        if is_in_progress_trial_set():
            trial = get_in_progress_trial()
            neps_state = get_workers_neps_state()
            root_directory = Path(neps_state.path)
            assert root_directory.exists()
            pipeline_directory = Path(trial.metadata.location)
            previous_pipeline_directory = (
                Path(trial.metadata.previous_trial_location)
                if trial.metadata.previous_trial_location
                else None
            )

        register_notify_trial_end("NEPS_TBLOGGER", cls.end_of_config)

        cls.config_working_directory = pipeline_directory
        cls.config_previous_directory = previous_pipeline_directory
        cls.optimizer_dir = root_directory

    @classmethod
    def WriteIncumbent(cls) -> None:  # noqa: N802
        """Enable logging of the incumbent (best) configuration for the current search."""
        cls.initiate_internal_configurations()
        cls.write_incumbent = True

    @classmethod
    def ConfigWriter(  # noqa: N802
        cls,
        *,
        write_summary_incumbent: bool = True,
        root_directory: Path | None = None,
        pipeline_directory: Path | None = None,
        previous_pipeline_directory: Path | None = None,
    ) -> SummaryWriter | None:
        """Create and return a TensorBoard SummaryWriter for NePS logging.

        Args:
            write_summary_incumbent (bool): Whether to write summaries for the incumbent.
            root_directory (Path | None): Root directory for NePS optimization.
            pipeline_directory (Path | None): Directory for current trial.
            previous_pipeline_directory (Path | None): Directory for previous trial.

        Returns:
            SummaryWriter | None: TensorBoard writer pointing to the NePS directory,
            or None if a writer cannot be initialized.
        """
        cls.write_incumbent = write_summary_incumbent
        cls.initiate_internal_configurations(
            root_directory,
            pipeline_directory,
            previous_pipeline_directory,
        )

        if (
            cls.config_previous_directory is None
            and cls.config_working_directory is not None
        ):
            cls.config_writer = SummaryWriter(cls.config_working_directory / "tbevents")
            return cls.config_writer

        if cls.config_working_directory is not None:
            init_dir = get_initial_directory(
                pipeline_directory=cls.config_working_directory,
            )
            if (init_dir / "tbevents").exists():
                cls.config_writer = SummaryWriter(init_dir / "tbevents")
                return cls.config_writer

            raise FileNotFoundError(
                "'tbevents' directory not found in the initial configuration directory."
            )

        return None

    @classmethod
    def end_of_config(cls, _: Trial) -> None:
        """Close the TensorBoard writer at the end of a configuration."""
        if cls.config_writer:
            cls.config_writer.close()
            cls.config_writer = None

        if cls.write_incumbent:
            cls._tracking_incumbent_api()

    @classmethod
    def _tracking_incumbent_api(cls) -> None:
        """Track the incumbent (best) configuration and log it to TensorBoard.

        Logs the best objective value over completed trials. Flushes and closes
        the writer to avoid conflicts in parallel execution.
        """
        assert cls.optimizer_dir is not None
        try:
            _, short = status(cls.optimizer_dir, print_summary=False)

            incum_tracker = short["num_success"] - 1
            incum_val = short["best_objective_to_minimize"]

            if cls.summary_writer is None and cls.optimizer_dir is not None:
                cls.summary_writer = SummaryWriter(cls.optimizer_dir / "summary_tb")

            assert cls.summary_writer is not None
            cls.summary_writer.add_scalar(
                tag="Summary/Incumbent_graph",
                scalar_value=incum_val,
                global_step=incum_tracker,
            )

            cls.summary_writer.flush()
            cls.summary_writer.close()
            time.sleep(0.5)

        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Incumbent tracking for TensorBoard failed and is now disabled: %s", e
            )
            cls.write_incumbent = False
