"""Plot results of a neural pipeline search run."""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

import numpy as np

from .plotting import _get_fig_and_axs, _map_axs, _plot_incumbent, _save_fig, _set_legend
from .read_results import process_seed


def plot(  # noqa: C901, PLR0913
    root_directory: str | Path,
    *,
    scientific_mode: bool = False,
    key_to_extract: str | None = None,
    benchmarks: list[str] | None = None,
    algorithms: list[str] | None = None,
    consider_continuations: bool = False,
    n_workers: int = 1,
    x_range: tuple | None = None,
    log_x: bool = False,
    log_y: bool = True,
    filename: str = "incumbent_trajectory",
    extension: str = "png",
    dpi: int = 100,
) -> None:
    """Plot results of a neural pipeline search run.

    Args:
        root_directory: The directory with neps results (see below).
        scientific_mode: If true, plot from a tree-structured root_directory:
            benchmark={}/algorithm={}/seed={}
        key_to_extract: The metric to be used on the x-axis
            (if active, make sure run_pipeline returns the metric in the info_dict)
        benchmarks: List of benchmarks to plot
        algorithms: List of algorithms to plot
        consider_continuations: If true, toggle calculation of continuation costs
        n_workers: Number of parallel processes of neps.run
        x_range: Bound x-axis (e.g. 1 10)
        log_x: If true, toggle logarithmic scale on the x-axis
        log_y: If true, toggle logarithmic scale on the y-axis
        filename: Filename
        extension: Image format
        dpi: Image resolution

    Raises:
        FileNotFoundError: If the data to be plotted is not present.
    """
    logger = logging.getLogger("neps")
    logger.info(f"Starting neps.plot using working directory {root_directory}")

    if benchmarks is None:
        benchmarks = ["example"]
    if algorithms is None:
        algorithms = ["neps"]

    logger.info(
        f"Processing {len(benchmarks)} benchmark(s) "
        f"and {len(algorithms)} algorithm(s)..."
    )

    ncols = 1 if len(benchmarks) == 1 else 2
    nrows = np.ceil(len(benchmarks) / ncols).astype(int)

    fig, axs = _get_fig_and_axs(nrows=nrows, ncols=ncols)

    base_path = Path(root_directory)

    for benchmark_idx, benchmark in enumerate(benchmarks):
        if scientific_mode:
            _base_path = base_path / f"benchmark={benchmark}"
            if not _base_path.is_dir():
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), _base_path
                )
        else:
            _base_path = None

        for algorithm in algorithms:
            seeds = [None]
            if _base_path is not None:
                assert scientific_mode
                _path = _base_path / f"algorithm={algorithm}"
                if not _path.is_dir():
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), _path
                    )

                seeds = sorted(os.listdir(_path))  # type: ignore
            else:
                _path = None

            incumbents = []
            costs = []
            max_costs = []
            for seed in seeds:
                incumbent, cost, max_cost = process_seed(
                    path=_path if _path is not None else base_path,
                    seed=seed,
                    key_to_extract=key_to_extract,
                    consider_continuations=consider_continuations,
                    n_workers=n_workers,
                )
                incumbents.append(incumbent)
                costs.append(cost)
                max_costs.append(max_cost)

            is_last_row = benchmark_idx >= (nrows - 1) * ncols
            is_first_column = benchmark_idx % ncols == 0
            xlabel = "Evaluations" if key_to_extract is None else key_to_extract.upper()
            _plot_incumbent(
                ax=_map_axs(
                    axs,
                    benchmark_idx,
                    len(benchmarks),
                    ncols,
                ),
                x=costs,
                y=incumbents,
                scale_x=max(max_costs) if key_to_extract == "fidelity" else None,
                title=benchmark if scientific_mode else None,
                xlabel=xlabel if is_last_row else None,
                ylabel="Best error" if is_first_column else None,
                log_x=log_x,
                log_y=log_y,
                x_range=x_range,
                label=algorithm,
            )

    if scientific_mode:
        _set_legend(
            fig,
            axs,
            benchmarks=benchmarks,
            algorithms=algorithms,
            nrows=nrows,
            ncols=ncols,
        )
    _save_fig(fig, output_dir=base_path, filename=filename, extension=extension, dpi=dpi)
    logger.info(f"Saved to '{base_path}/{filename}.{extension}'")
