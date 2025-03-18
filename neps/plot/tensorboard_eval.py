"""The tblogger module provides a simplified interface for logging to TensorBoard."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from torch.utils.tensorboard.writer import SummaryWriter

from neps.runtime import (
    get_in_progress_trial,
    get_workers_neps_state,
    register_notify_trial_end,
)
from neps.status.status import status
from neps.utils.common import get_initial_directory

if TYPE_CHECKING:
    from neps.state.trial import Trial

logger = logging.getLogger(__name__)


class tblogger:  # noqa: N801
    """The tblogger class provides a simplified interface for logging to tensorboard."""

    config_id: ClassVar[str | None] = None
    config: ClassVar[Mapping[str, Any] | None] = None
    config_working_directory: ClassVar[Path | None] = None
    optimizer_dir: ClassVar[Path | None] = None
    config_previous_directory: ClassVar[Path | None] = None

    write_incumbent: ClassVar[bool | None] = None

    config_writer: ClassVar[SummaryWriter | None] = None
    summary_writer: ClassVar[SummaryWriter | None] = None

    @staticmethod
    def _initiate_internal_configurations() -> None:
        """Track the Configuration space data from the way handled by neps runtime
        '_sample_config' to keep in sync with config ids and directories NePS is
        operating on.
        """
        trial = get_in_progress_trial()
        neps_state = get_workers_neps_state()

        register_notify_trial_end("NEPS_TBLOGGER", tblogger.end_of_config)

        # We are assuming that neps state is all filebased here
        root_dir = Path(neps_state.path)
        assert root_dir.exists()

        tblogger.config_working_directory = Path(trial.metadata.location)
        tblogger.config_previous_directory = (
            Path(trial.metadata.previous_trial_location)
            if trial.metadata.previous_trial_location is not None
            else None
        )
        tblogger.config_id = trial.metadata.id
        tblogger.optimizer_dir = root_dir
        tblogger.config = trial.config

    @staticmethod
    def ConfigWriter(*, write_summary_incumbent: bool = True) -> SummaryWriter:  # noqa: N802
        """Creates and returns a TensorBoard SummaryWriter configured to write logs
        to the appropriate directory for NePS.

        Args:
            write_summary_incumbent (bool): Determines whether to write summaries
                                            for the incumbent configurations.
                                            Defaults to True.

        Returns:
            SummaryWriter: An instance of TensorBoard SummaryWriter pointing to the
                        designated NePS directory.
        """
        tblogger.write_incumbent = write_summary_incumbent
        tblogger._initiate_internal_configurations()
        # This code runs only once per config, to assign that config a config_writer.
        if (
            tblogger.config_previous_directory is None
            and tblogger.config_working_directory is not None
        ):
            # If no fidelities are there yet, define the writer via the config_id
            tblogger.config_id = str(tblogger.config_working_directory).rsplit(
                "/", maxsplit=1
            )[-1]
            tblogger.config_writer = SummaryWriter(
                tblogger.config_working_directory / "tbevents"
            )
            return tblogger.config_writer

        # Searching for the initial directory where tensorboard events are stored.
        if tblogger.config_working_directory is not None:
            init_dir = get_initial_directory(
                pipeline_directory=tblogger.config_working_directory
            )
            tblogger.config_id = str(init_dir).rsplit("/", maxsplit=1)[-1]
            if (init_dir / "tbevents").exists():
                tblogger.config_writer = SummaryWriter(init_dir / "tbevents")
                return tblogger.config_writer

            raise FileNotFoundError(
                "'tbevents' was not found in the initial directory of the configuration."
            )
        return None

    @staticmethod
    def end_of_config(trial: Trial) -> None:  # noqa: ARG004
        """Closes the writer."""
        if tblogger.config_writer:
            # Close and reset previous config writers for consistent logging.
            # Prevent conflicts by reinitializing writers when logging ongoing.
            tblogger.config_writer.close()
            tblogger.config_writer = None

        if tblogger.write_incumbent:
            tblogger._tracking_incumbent_api()

    @staticmethod
    def _tracking_incumbent_api() -> None:
        """Track the incumbent (best) objective_to_minimize and log it in the TensorBoard
            summary.

        Note:
            The function relies on the following global variables:
                - tblogger.optimizer_dir
                - tblogger.summary_writer

            The function logs the incumbent trajectory in TensorBoard.
        """
        assert tblogger.optimizer_dir is not None
        try:
            _, short = status(tblogger.optimizer_dir, print_summary=False)

            incum_tracker = short["num_success"] - 1
            incum_val = short["best_objective_to_minimize"]

            if tblogger.summary_writer is None and tblogger.optimizer_dir is not None:
                tblogger.summary_writer = SummaryWriter(
                    tblogger.optimizer_dir / "summary"
                )

            assert tblogger.summary_writer is not None
            tblogger.summary_writer.add_scalar(
                tag="Summary/Incumbent_graph",
                scalar_value=incum_val,
                global_step=incum_tracker,
            )

            # Frequent writer open/close creates new 'tfevent' files due to
            # parallelization needs. Simultaneous open writers risk conflicts,
            # so they're flushed and closed after use.

            tblogger.summary_writer.flush()
            tblogger.summary_writer.close()
            time.sleep(0.5)

        except ValueError as e:
            logger.critical(
                "Incumbent tracking for TensorBoard with NePS has failed due to "
                f"a ValueError: {e}. This feature is now permanently disabled"
                " for the entire run."
            )
            tblogger.write_incumbent = False
