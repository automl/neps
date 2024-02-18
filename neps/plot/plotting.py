from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

map_axs = (
    lambda axs, idx, length, ncols: axs
    if length == 1
    else (axs[idx] if length == ncols else axs[idx // ncols][idx % ncols])
)


def _set_general_plot_style() -> None:
    """
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("deep")
    """
    # plt.switch_backend("pgf")
    plt.rcParams.update(
        {
            "text.usetex": False,  # True,
            # "pgf.texsystem": "pdflatex",
            # "pgf.rcfonts": False,
            # "font.family": "serif",
            # "font.serif": [],
            # "font.sans-serif": [],
            # "font.monospace": [],
            "font.size": "10.90",
            "legend.fontsize": "9.90",
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
            "legend.title_fontsize": "small",
            # "bottomlabel.weight": "normal",
            # "toplabel.weight": "normal",
            # "leftlabel.weight": "normal",
            # "tick.labelweight": "normal",
            # "title.weight": "normal",
            # "pgf.preamble": r"""
            #    \usepackage[T1]{fontenc}
            #    \usepackage[utf8x]{inputenc}
            #    \usepackage{microtype}
            # """,
        }
    )


def get_fig_and_axs(
    nrows: int = 1,
    ncols: int = 1,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    _set_general_plot_style()

    figsize = (4 * ncols, 3 * nrows)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
    )

    fig.tight_layout(pad=2.0, h_pad=2.5)
    sns.despine(fig)

    return fig, axs


def plot_incumbent(
    ax: matplotlib.axes.Axes,
    x: list | np.ndarray,
    y: list | np.ndarray,
    scale_x: float | None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    x_range: tuple | None = None,
    **plotting_kwargs,
) -> None:
    df = interpolate_time(incumbents=y, costs=x, x_range=x_range, scale_x=scale_x)
    df = df_to_x_range(df, x_range=x_range)

    x = df.index
    y_mean = df.mean(axis=1).values
    ddof = 0 if len(df.columns) == 1 else 1
    std_error = stats.sem(df.values, axis=1, ddof=ddof)

    ax.plot(x, y_mean, linestyle="-", linewidth=0.7, **plotting_kwargs)

    ax.fill_between(
        x,
        y_mean - std_error,
        y_mean + std_error,
        # color=COLOR_MARKER_DICT[algorithm],
        alpha=0.2,
    )

    ax.set_xlim(auto=True)

    if title is not None:
        ax.set_title(title, fontsize=20)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18, color=(0, 0, 0, 0.69))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18, color=(0, 0, 0, 0.69))
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("symlog")
    if x_range is not None:
        ax.set_xlim(*x_range)
    ax.set_ylim(auto=True)

    # Black with some alpha
    ax.tick_params(axis="both", which="major", labelsize=18, labelcolor=(0, 0, 0, 0.69))
    ax.grid(True, which="both", ls="-", alpha=0.8)


def interpolate_time(
    incumbents: list | np.ndarray,
    costs: list | np.ndarray,
    x_range: tuple | None = None,
    scale_x: float | None = None,
) -> pd.DataFrame:
    if isinstance(incumbents, list):
        incumbents = np.array(incumbents)
    if isinstance(costs, list):
        costs = np.array(costs)

    df_dict = {}

    for i, _ in enumerate(incumbents):
        _seed_info = pd.Series(incumbents[i], index=np.cumsum(costs[i]))
        df_dict[f"seed{i}"] = _seed_info
    df = pd.DataFrame.from_dict(df_dict)

    # important step to plot func evals on x-axis
    df.index = df.index if scale_x is None else df.index.values / scale_x

    if x_range is not None:
        min_b, max_b = x_range
        new_entry = {c: np.nan for c in df.columns}
        _df = pd.DataFrame.from_dict(new_entry, orient="index").T
        _df.index = [min_b]
        df = pd.concat((df, _df)).sort_index()
        new_entry = {c: np.nan for c in df.columns}
        _df = pd.DataFrame.from_dict(new_entry, orient="index").T
        _df.index = [max_b]
        df = pd.concat((df, _df)).sort_index()

    df = df.fillna(method="backfill", axis=0).fillna(method="ffill", axis=0)
    if x_range is not None:
        df = df.query(f"{x_range[0]} <= index <= {x_range[1]}")

    return df


def df_to_x_range(df: pd.DataFrame, x_range: tuple | None = None) -> pd.DataFrame:
    x_max = np.inf if x_range is None else int(x_range[-1])
    new_entry = {c: np.nan for c in df.columns}
    _df = pd.DataFrame.from_dict(new_entry, orient="index").T
    _df.index = [x_max]
    df = pd.concat((df, _df)).sort_index()
    df = df.fillna(method="backfill", axis=0).fillna(method="ffill", axis=0)

    return df


def set_legend(
    fig: matplotlib.figure.Figure,
    axs: matplotlib.axes.Axes,
    benchmarks: list[str],
    algorithms: list[str],
    nrows: int,
    ncols: int,
) -> None:
    bbox_y_mapping = {
        1: -0.22,
        2: -0.11,
        3: -0.07,
        4: -0.05,
        5: -0.04,
    }
    anchor_y = bbox_y_mapping[nrows]
    bbox_to_anchor = (0.5, anchor_y)

    handles, labels = map_axs(axs, 0, len(benchmarks), ncols).get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        fontsize="large",
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=len(algorithms),
        frameon=True,
    )

    for legend_item in legend.legendHandles:
        legend_item.set_linewidth(2.0)


def save_fig(
    fig: matplotlib.figure.Figure,
    output_dir: Path | str,
    filename: str = "incumbent_trajectory",
    extension: str = "png",
    dpi: int = 100,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{filename}.{extension}",
        bbox_inches="tight",
        dpi=dpi,
    )
