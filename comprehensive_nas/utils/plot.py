from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from path import Path
from scipy import stats

color_marker_dict = {
    "random_search": {"color": "red", "marker": "o"},
    "bayes_opt": {"color": "green", "marker": "o"},
    "DEHB": {"color": "deeppink", "marker": "o"},
}


class PlotTypes(IntEnum):
    REGRET_CUM_EVAL_COST = 1
    INCUMBENT_CUM_EVAL_COST = 2
    INCUMBENT_RUNTIME = 3

    @staticmethod
    def get_plot_X_Y_title(title_type):
        if title_type == PlotTypes.REGRET_CUM_EVAL_COST:
            return "Cumulative Eval Cost", "Simple Regret"
        elif title_type == PlotTypes.INCUMBENT_RUNTIME:
            return "Runtime (s)", "Incumbent"
        elif title_type == PlotTypes.INCUMBENT_CUM_EVAL_COST:
            return "Cumulative Eval Cost", "Incumbent"

    @staticmethod
    def get_plot_X_Y_keys(title_type):
        if title_type == PlotTypes.REGRET_CUM_EVAL_COST:
            return "cum_eval_cost", "simple_regret_incumbents"
        elif title_type == PlotTypes.INCUMBENT_RUNTIME:
            return "runtime", "incumbent_values"
        elif title_type == PlotTypes.INCUMBENT_CUM_EVAL_COST:
            return "cum_eval_cost", "incumbent_values"


STRING_TO_PLOTTYPE = {
    "regret_cost": PlotTypes.REGRET_CUM_EVAL_COST,
    "incumbent_cost": PlotTypes.INCUMBENT_CUM_EVAL_COST,
    "incumbent_time": PlotTypes.INCUMBENT_RUNTIME,
}


PLOTTYPE_TO_STRING = {
    PlotTypes.REGRET_CUM_EVAL_COST: "regret_cost",
    PlotTypes.INCUMBENT_CUM_EVAL_COST: "incumbent_cost",
    PlotTypes.INCUMBENT_RUNTIME: "incumbent_time",
}


def make_incumbent_plot(
    strategy_results_dict,
    strategies,
    save_folder_name,
    benchmark_name,
    title_type,
    experiment_name=None,
):
    trajectories = get_trajectories(strategy_results_dict, strategies, title_type)
    fig, ax = plt.subplots(1, figsize=(6, 4))
    plot_losses(fig, ax, None, trajectories, regret=False, plot_mean=True)

    X_label, y_label = PlotTypes.get_plot_X_Y_title(title_type)
    ax.set_xlabel(X_label)
    ax.set_ylabel(y_label)
    if title_type == PlotTypes.REGRET_CUM_EVAL_COST:
        ax.set_yscale("log")
    elif title_type == PlotTypes.INCUMBENT_RUNTIME:
        ax.set_xscale("log")
    plt.legend()
    plt.title(benchmark_name)
    plt.grid(True, which="both", ls="-", alpha=0.8)
    plt.tight_layout()

    plt.savefig("{}/{}.png".format(save_folder_name, experiment_name))


def get_trajectories(strategy_dict, strategies, title_type):
    all_trajectories = {}
    X_key, y_key = PlotTypes.get_plot_X_Y_keys(title_type)
    for strategy in strategies:
        dfs = []
        data = strategy_dict[strategy]
        losses = data[y_key]
        times = data[X_key]
        for i, loss in enumerate(losses):
            time = times[i][-len(loss) :]
            print("Seed: ", i, " MIN: ", min(loss))

            df = pd.DataFrame({str(i): loss}, index=time)
            dfs.append(df)

        df = merge_and_fill_trajectories(dfs, default_value=None)
        if df.empty:
            continue

        all_trajectories[strategy] = {
            "time_stamps": np.array(df.index),
            "losses": np.array(df.T),
        }

    return all_trajectories


def merge_and_fill_trajectories(pandas_data_frames, default_value=None):
    # merge all tracjectories keeping all time steps
    df = pd.DataFrame().join(pandas_data_frames, how="outer")

    # forward fill to make it a proper step function
    df = df.fillna(method="ffill")

    if default_value is None:
        # backward fill to replace the NaNs for the early times by
        # the performance of a random configuration
        df = df.fillna(method="bfill")
    else:
        df = df.fillna(default_value)

    return df


def plot_losses(
    fig,
    ax,
    axins,
    incumbent_trajectories,
    regret=True,
    incumbent=None,
    show=True,
    linewidth=1,
    marker_size=3,
    xscale="log",
    xlabel="wall clock time [s]",
    yscale="log",
    ylabel=None,
    legend_loc="best",
    xlim=None,
    ylim=None,
    plot_mean=True,
    labels={},
    figsize=(16, 9),
):
    if regret:
        if ylabel is None:
            ylabel = "regret"
        # find lowest performance in the data to update incumbent

        if incumbent is None:
            incumbent = np.inf
            for tr in incumbent_trajectories.values():
                incumbent = min(tr["losses"][:, -1].min(), incumbent)
            print("incumbent value: ", incumbent)

    for _, (m, tr) in enumerate(incumbent_trajectories.items()):
        trajectory = np.copy(tr["losses"])
        if trajectory.shape[0] == 0:
            continue
        if regret:
            trajectory -= incumbent

        sem = np.sqrt(trajectory.var(axis=0, ddof=1) / tr["losses"].shape[0])
        if plot_mean:
            mean = trajectory.mean(axis=0)
        else:
            mean = np.median(trajectory, axis=0)
            sem *= 1.253
        len_tr = trajectory.shape[1]
        errorevery = int(0.15 * len_tr)

        ax.errorbar(
            tr["time_stamps"],
            mean,
            sem,
            uplims=True,
            lolims=True,
            errorevery=errorevery,
            color=color_marker_dict[m]["color"],
            alpha=0.4,
        )

        ax.plot(
            tr["time_stamps"],
            mean,
            label=labels.get(m, m),
            color=color_marker_dict[m]["color"],
            linewidth=linewidth,
            marker=color_marker_dict[m]["marker"],
            markersize=marker_size,
            markevery=(0.1, 0.1),
        )

        if axins is not None:
            axins.plot(
                tr["time_stamps"],
                mean,
                label=labels.get(m, m),
                color=color_marker_dict[m]["color"],
                linewidth=linewidth,
                marker=color_marker_dict[m]["marker"],
                markersize=marker_size,
                markevery=(0.1, 0.1),
            )

    return fig, ax


def plot_incumbent_trajectory(
    x: list,
    incumbents: dict,
    save_path: Path,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    plot_mean: bool = True,
    plot_std_error: bool = True,
):
    for strategy, vals in sorted(incumbents.items()):
        vals = np.array(vals)
        if len(vals.shape) == 1:
            plt.plot(x, vals, label=strategy)
        elif len(vals.shape) == 2:
            axis = 0 if len(x) == vals.shape[1] else 1
            y = np.mean(vals, axis=axis) if plot_mean else np.median(vals, axis=axis)
            if vals.shape[0] == 1:
                print(f"WARNING: {strategy} has only one run!")
                y_err = np.zeros_like(x)
            else:
                y_err = (
                    stats.sem(vals, axis=axis)
                    if plot_std_error
                    else np.std(vals, axis=axis)
                )
            plt.plot(x, y, "x-", label=strategy)
            plt.fill_between(x, y - y_err, y + y_err, alpha=0.4)
        else:
            raise ValueError(
                "Plot incumbent trajectory only supports 1- or 2-dimensional values"
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title if title is not None else "Incumbent trajectory")
    plt.grid(True, which="both", ls="-", alpha=0.8)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
