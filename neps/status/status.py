"""Functions to get the status of a run and save the status to CSV files."""

# ruff: noqa: T201
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from neps.runtime import get_workers_neps_state
from neps.space.neps_spaces import neps_space
from neps.space.neps_spaces.neps_space import NepsCompatConverter
from neps.space.neps_spaces.sampling import OnlyPredefinedValuesSampler
from neps.state.neps_state import FileLocker, NePSState
from neps.state.trial import State, Trial

if TYPE_CHECKING:
    from neps.space.neps_spaces.parameters import PipelineSpace


@dataclass
class Summary:
    """Summary of the current state of a neps run."""

    by_state: dict[State, list[Trial]]
    # No support for multiobjective yet
    best: tuple[Trial, float] | None
    is_multiobjective: bool
    running: list[Trial] = field(init=False)

    def __post_init__(self) -> None:
        if self.is_multiobjective:
            assert self.best is None

        self.running = self.by_state[State.EVALUATING]

    @property
    def num_errors(self) -> int:
        """Number of trials that have errored."""
        return len(self.by_state[State.CRASHED])

    def df(self) -> pd.DataFrame:
        """Convert the summary into a dataframe."""
        trials = sorted(
            itertools.chain(*self.by_state.values()),
            key=lambda t: t.metadata.time_sampled,
        )

        # Config dataframe, config columns prefixed with `config.`
        config_df = (
            pd.DataFrame.from_records([trial.config for trial in trials])
            .rename(columns=lambda name: f"config.{name}")
            .convert_dtypes()
        )

        # Report dataframe
        report_df = pd.DataFrame.from_records(
            [asdict(t.report) if t.report is not None else {} for t in trials]
        ).convert_dtypes()

        extra_df = pd.DataFrame()
        # We pop out the user extra column to flatten it
        if "extra" in report_df.columns:
            extra_column = report_df.pop("extra")
            extra_df = pd.json_normalize(extra_column).rename(  # type: ignore
                columns=lambda name: f"extra.{name}"
            )

        # Metadata dataframe
        metadata_df = pd.DataFrame.from_records(
            [asdict(t.metadata) for t in trials]
        ).convert_dtypes()

        return (
            pd.concat([config_df, extra_df, report_df, metadata_df], axis="columns")
            .set_index("id")
            .dropna(how="all", axis="columns")
        )

    def completed(self) -> list[Trial]:
        """Return all trials which are in a completed state."""
        return list(
            itertools.chain(
                self.by_state[State.SUCCESS],
                self.by_state[State.FAILED],
                self.by_state[State.CRASHED],
            )
        )

    @property
    def num_evaluated(self) -> int:
        """Number of trials that have been evaluated."""
        return (
            len(self.by_state[State.SUCCESS])
            + len(self.by_state[State.FAILED])
            + len(self.by_state[State.CRASHED])
        )

    @property
    def num_pending(self) -> int:
        """Number of trials that are pending."""
        return len(self.by_state[State.PENDING])

    def formatted(  # noqa: PLR0912, C901
        self, pipeline_space_variables: tuple[PipelineSpace, list[str]] | None = None
    ) -> str:
        """Return a formatted string of the summary.

        Args:
            pipeline_space_variables: If provided, this tuple contains the Pipeline and a
                list of variable names to format the config in the summary. This is useful
                for pipelines that have a complex configuration structure, allowing for a
                more readable output.

                !!! Warning:

                    This is only supported when using NePS-only optimizers, such as
                    `neps.algorithms.neps_random_search`,
                    `neps.algorithms.complex_random_search`
                    or `neps.algorithms.neps_priorband`. When the search space is
                    simple enough, using `neps.algorithms.random_search` or
                    `neps.algorithms.priorband` is not enough, as it will be transformed
                    to a simpler HPO framework, which is incompatible with the
                    `pipeline_space_variables` argument.

        Returns:
            A formatted string of the summary.
        """
        state_summary = "\n".join(
            f"    {state.name.lower()}: {len(trials)}"
            for state, trials in self.by_state.items()
            if len(trials) > 0
        )

        if self.best is None:
            if self.is_multiobjective:
                best_summary = "Multiobjective summary not supported yet for best yet."
            else:
                best_summary = "No best found yet."
        else:
            best_trial, best_objective_to_minimize = self.best

            # Format config based on whether pipeline_space_variables is provided

            best_summary = (
                f"# Best Found (config {best_trial.metadata.id}):"
                "\n"
                f"\n    objective_to_minimize: {best_objective_to_minimize}"
            )
            if pipeline_space_variables is None:
                best_summary += f"\n    config: {best_trial.config}"
            else:
                best_config_resolve = NepsCompatConverter().from_neps_config(
                    best_trial.config
                )
                pipeline_configs = []
                for variable in pipeline_space_variables[1]:
                    pipeline_configs.append(
                        neps_space.config_string.ConfigString(
                            neps_space.convert_operation_to_string(
                                getattr(
                                    neps_space.resolve(
                                        pipeline_space_variables[0],
                                        OnlyPredefinedValuesSampler(
                                            best_config_resolve.predefined_samplings
                                        ),
                                        environment_values=best_config_resolve.environment_values,
                                    )[0],
                                    variable,
                                )
                            )
                        ).pretty_format()
                    )

                for n_pipeline, pipeline_config in enumerate(pipeline_configs):
                    if isinstance(pipeline_config, str):
                        # Replace literal \t and \n with actual formatting
                        formatted_config = pipeline_config.replace("\\t", "    ").replace(
                            "\\n", "\n"
                        )

                        # Add proper indentation to each line
                        lines = formatted_config.split("\n")
                        indented_lines = []
                        for i, line in enumerate(lines):
                            if i == 0:
                                indented_lines.append(
                                    line
                                )  # First line gets base indentation
                            else:
                                indented_lines.append(
                                    "        " + line
                                )  # Subsequent lines get extra indentation

                        formatted_config = "\n".join(indented_lines)
                    else:
                        formatted_config = pipeline_config  # type: ignore
                    best_summary += (
                        f"\n    config: {pipeline_space_variables[1][n_pipeline]}\n      "
                        f"  {formatted_config}"
                    )

            best_summary += f"\n    path: {best_trial.metadata.location}"

            assert best_trial.report is not None
            if best_trial.report.cost is not None:
                best_summary += f"\n    cost: {best_trial.report.cost}"
            if len(best_trial.report.extra) > 0:
                best_summary += f"\n    extra: {best_trial.report.extra}"

        return f"# Configs: {self.num_evaluated}\n\n{state_summary}\n\n{best_summary}"

    @classmethod
    def from_directory(cls, root_directory: str | Path) -> Summary:
        """Create a summary from a neps run directory."""
        root_directory = Path(root_directory)

        is_multiobjective: bool = False
        best: tuple[Trial, float] | None = None
        by_state: dict[State, list[Trial]] = {s: [] for s in State}

        # NOTE: We don't lock the shared state since we are just reading and don't need to
        # make decisions based on the state
        try:
            shared_state = get_workers_neps_state()
        except RuntimeError:
            shared_state = NePSState.create_or_load(root_directory, load_only=True)

        trials = shared_state.lock_and_read_trials()

        for _trial_id, trial in trials.items():
            state = trial.metadata.state
            by_state[state].append(trial)

            if trial.report is not None:
                objective_to_minimize = trial.report.objective_to_minimize
                match objective_to_minimize:
                    case None:
                        pass
                    case float() | int() | np.number() if not is_multiobjective:
                        if best is None or objective_to_minimize < best[1]:
                            best = (trial, objective_to_minimize)
                    case Sequence():
                        is_multiobjective = True
                        best = None
                    case _:
                        raise RuntimeError("Unexpected type for objective_to_minimize")

        return cls(by_state=by_state, best=best, is_multiobjective=is_multiobjective)


def status(
    root_directory: str | Path,
    *,
    print_summary: bool = False,
    pipeline_space_variables: tuple[PipelineSpace, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Print status information of a neps run and return results.

    Args:
        root_directory: The root directory given to neps.run.
        print_summary: If true, print a summary of the current run state.
        pipeline_space_variables: If provided, this tuple contains the Pipeline and a
            list of variable names to format the config in the summary. This is useful
            for pipelines that have a complex configuration structure, allowing for a
            more readable output.

            !!! Warning:

                This is only supported when using NePS-only optimizers, such as
                `neps.algorithms.neps_random_search`,
                `neps.algorithms.complex_random_search`
                or `neps.algorithms.neps_priorband`. When the search space is
                simple enough, using `neps.algorithms.random_search` or
                `neps.algorithms.priorband` is not enough, as it will be transformed to a
                simpler HPO framework, which is incompatible with the
                `pipeline_space_variables` argument.

    Returns:
        Dataframe of full results and short summary series.
    """
    root_directory = Path(root_directory)
    summary = Summary.from_directory(root_directory)

    if print_summary:
        print(summary.formatted(pipeline_space_variables=pipeline_space_variables))

    df = summary.df()

    if len(df) == 0:
        return df, pd.Series()

    short = (
        df.groupby("state")
        .size()
        .rename(lambda name: f"num_{name.replace('State.', '').lower()}")
    )
    short.name = "value"
    short.index.name = "summary"
    short.index = short.index.astype(str)
    assert isinstance(short, pd.Series)

    # Not implemented for hypervolume -_-
    if summary.is_multiobjective:
        return df, short

    idx_min = df["objective_to_minimize"].idxmin()
    row = df.loc[idx_min]
    assert isinstance(row, pd.Series)
    short["best_objective_to_minimize"] = row["objective_to_minimize"]
    short["best_config_id"] = row.name

    row = row.loc[row.index.str.startswith("config.")]
    row.index = row.index.str.replace("config.", "")  # type: ignore
    short = pd.concat([short, row])  # type: ignore
    assert isinstance(short, pd.Series)
    return df, short


def _initiate_summary_csv(root_directory: str | Path) -> tuple[Path, Path, FileLocker]:
    """Initializes a summary CSV and an associated locker for file access control.

    Args:
        root_directory: The root directory where the summary CSV directory,
            containing CSV files and a locker for file access control, will be created.

    Returns:
        A tuple containing the file paths for the configuration data CSV, run data CSV,
        and a locker file.

    The locker is used for file access control to ensure data integrity in a
    multi-threaded or multi-process environment.
    """
    root_directory = Path(root_directory).absolute().resolve()
    summary_path = root_directory / "summary"
    summary_path.mkdir(parents=True, exist_ok=True)

    full_path = summary_path / "full.csv"
    short_path = summary_path / "short.csv"
    csv_locker = FileLocker(summary_path / ".csv_lock", poll=0.1, timeout=10)

    return (full_path, short_path, csv_locker)


def post_run_csv(root_directory: str | Path) -> tuple[Path, Path]:
    """Create CSV files summarizing the run data.

    Args:
        root_directory: The root directory of the NePS run.

    Returns:
        The paths to the configuration data CSV and the run data CSV.
    """
    full_df, short = status(root_directory, print_summary=False)
    full_df_path, short_path, csv_locker = _initiate_summary_csv(root_directory)

    with csv_locker.lock():
        full_df.to_csv(full_df_path)
        short.to_frame().to_csv(short_path)

    return full_df_path, short_path
