"""Scaling plots and predictions, with the CSVs of their points.

The pre-train loss goes to `summary/pre_train/` and each downstream performance to
`summary/downstream/<metric>/`.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import portalocker as pl
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from scalefm.optimizers.optimizer import Artifact, ArtifactType
from scalefm.state.filebased import FileLocker
from scalefm.utils.pareto import pareto_efficient

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scalefm.prediction import Target
    from scalefm.space import SearchSpace
    from scalefm.state.trial import Trial

logger = logging.getLogger(__name__)

PARETO_COLOR = "#D55E00"
DOMINATED_COLOR = "#B0B0B0"
INK = "#2B2B2B"
SECONDARY_INK = "#52514E"
AXIS_COLOR = "#C3C2B7"
GRID_COLOR = "#E6E5DF"
EXTRAPOLATION_COLOR = "#F3F2EE"
TARGET_COLORS = ["#86B6EF", "#6DA7EC", "#5598E7", "#3987E5", "#2A78D6", "#256ABF"]
COLUMNS = ["trial_id", "cost", "loss", "on_frontier"]
DEFAULT_TARGETS = [10, 100]
TENSORBOARD_LOCK_TIMEOUT = 5.0
"""Seconds to wait for another worker logging to TensorBoard, before trying again at
the next update."""
NUM_FIT_POINTS = 60
"""Costs the fit is drawn at, between the cheapest run and the largest target."""
NUM_GP_FIT_POINTS = 15
"""The same for the GP posterior, which takes a search over configs per cost."""


@dataclass
class Estimators:
    """The parameter count and cost of a config, as given to `scalefm.predict()`."""

    params: Callable[..., float] | None = None
    cost: Callable[..., float] | None = None


def _law(predictor: str, cost: str, estimators: Estimators | None) -> str:
    """The scaling law to fit over `cost`: `predictor` over the training cost, if it
    can be fitted, and a power law otherwise. The GP posterior needs a cost estimator.
    """
    if cost != "cost":
        return "power_law"
    if predictor == "gp_posterior" and (estimators is None or estimators.cost is None):
        return "power_law"
    return predictor


def _is_higher_better(metric: str) -> bool:
    """Accuracy-like metrics, which are flipped to `1 - value`."""
    return "_acc" in metric or "recall" in metric


def _downstream(performances: dict[str, float]) -> dict[str, float]:
    """The downstream performances of a report, as `downstream/<name>` columns."""
    metrics = {}
    for metric, value in performances.items():
        flipped = 1.0 - value if _is_higher_better(metric) else value
        metrics[f"downstream/{metric}"] = flipped
    return metrics


def _own_units(metric: str, values: pd.Series) -> pd.Series:
    """The values of a metric column, flipped back for accuracy-like metrics."""
    return 1.0 - values if _is_higher_better(metric) else values


def _name(metric: str) -> str:
    """The name of a metric column in file names, e.g. `val_acc`."""
    return metric.removeprefix("downstream/").replace(".", "_")


def _label(metric: str) -> str:
    """The axis label of a metric column."""
    name = metric.removeprefix("downstream/")
    if metric == "loss":
        return "Loss"
    return f"1 - {name}" if _is_higher_better(name) else name


def _costs(table: pd.DataFrame) -> list[str]:
    """`cost`, and `inference_cost` if the runs report it."""
    return ["cost", "inference_cost"] if "inference_cost" in table.columns else ["cost"]


def _targets(
    table: pd.DataFrame, cost: str, targets: Mapping[str, Sequence[float]] | None
) -> list[float]:
    """The target costs given for `cost`, or `DEFAULT_TARGETS` times the largest."""
    if targets and cost in targets:
        return [float(target) for target in targets[cost]]
    return [factor * float(table[cost].max()) for factor in DEFAULT_TARGETS]


def _with_frontier(table: pd.DataFrame, cost: str, loss: str) -> pd.DataFrame:
    """The runs that report `cost` and `loss`, with whether each is on their frontier."""
    table = table.dropna(subset=[cost, loss])
    return table.assign(on_frontier=pareto_efficient(table[[cost, loss]].to_numpy(float)))


def runs_table(trials: Sequence[Trial]) -> pd.DataFrame:
    """One row per successful trial with a positive cost: its cost, loss, whether it is
    on the Pareto frontier, the numbers of its config and `info_dict`, its inference
    cost, downstream performances, model size `n_params` and data `n_data`.
    """
    rows: list[dict] = []
    for trial in trials:
        report = trial.report
        if report is None or report.reported_as != "success":
            continue
        if report.cost is None or not report.cost > 0:
            continue
        extra = dict(report.extra or {})
        row: dict = {}
        for values in (extra, trial.config):
            for name, value in values.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    row[name] = value
        row.update(_downstream(report.downstream_performances))
        if report.inference_cost is not None:
            row["inference_cost"] = report.inference_cost
        row["trial_id"] = trial.id
        row["cost"] = float(report.cost)
        row["loss"] = float(report.objective_to_minimize)  # type: ignore[arg-type]
        row["n_params"] = report.n_params
        row["n_data"] = report.n_data
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table["on_frontier"] = pareto_efficient(table[["cost", "loss"]].to_numpy(float))
    return table


def predict_over(
    runs: pd.DataFrame,
    metric: str,
    cost: str,
    law: str,
    costs: Sequence[float],
    targets: Sequence[Target],
    space: SearchSpace | None = None,
    estimators: Estimators | None = None,
) -> pd.DataFrame:
    """What the scaling law `law`, fitted on `metric` over `cost` of `runs`, predicts
    for `targets` at each of `costs`: the metric, the model size and data, and with
    the search space `space` the config. What it cannot predict is left out.
    `estimators` are the parameter count and cost of a config, which the GP posterior
    needs.

    Over the inference cost, the data is the most any run used, since more data does
    not make a model more expensive to run.
    """
    from scalefm.prediction import Target, fit_and_predict, get_predictor_type

    rows = list(
        zip(runs["n_params"], runs["n_data"], runs[metric], runs[cost], strict=True)
    )
    names = [] if space is None else list(space.searchables)
    configs = runs.reindex(columns=names).to_dict("records")
    predicted = pd.DataFrame({cost: costs})
    for target in targets:
        try:
            predictions = fit_and_predict(
                get_predictor_type(law),
                [float(c) for c in costs],
                targets=[target],
                rows=rows,
                configs=configs,
                space=space,
                params_estimator=None if estimators is None else estimators.params,
                cost_estimator=None if estimators is None else estimators.cost,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not fit '{law}' on {metric} over {cost}: {e}")
            continue
        if target == Target.LOSS:
            predicted[metric] = [p.loss for p in predictions]
        elif target == Target.COARSE:
            predicted["n_params"] = [p.n_params for p in predictions]
            predicted["n_data"] = [p.n_data for p in predictions]
            if cost != "cost":
                predicted["n_data"] = runs["n_data"].max()
        else:
            for name in names:
                predicted[name] = [p.config[name] for p in predictions]  # type: ignore[index]
    return predicted


def _draw_fit(
    ax: Axes,
    fit: pd.DataFrame,
    cost: str,
    column: str,
    largest_run: float,
    label: str,
) -> None:
    """The fit of `column`, solid over the runs and dashed beyond the largest one, if
    it is drawn there, its predictions at the target costs, and the extrapolation
    region shaded.
    """
    line = fit[~fit["target"]]
    inside = line[line[cost] <= largest_run]
    beyond = line[line[cost] >= inside[cost].max()]
    targets = fit[fit["target"]]
    ax.plot(inside[cost], inside[column], color=INK, linewidth=2, label=label)
    if len(beyond) > 1:
        ax.plot(
            beyond[cost], beyond[column], color=INK, linewidth=2, linestyle=(0, (4, 3))
        )
    ax.scatter(
        targets[cost],
        targets[column],
        s=90,
        c=_target_colors(len(targets)),
        edgecolors="white",
        linewidths=1.5,
        zorder=4,
        label="Prediction",
    )
    ax.set(xscale="log", yscale="log")
    xmin, xmax = ax.get_xlim()
    ax.axvspan(
        largest_run,
        xmax,
        color=EXTRAPOLATION_COLOR,
        zorder=0,
        label="Extrapolation region",
    )
    ax.set_xlim(xmin, xmax)


def plot_loss_over_cost(
    runs: pd.DataFrame, loss: str, cost: str, fit: pd.DataFrame, label: str
) -> Figure:
    """Every run's `loss` over its `cost`, their Pareto frontier, and the fit `label`
    extended to the target costs, with its predictions there.
    """
    front = runs[runs["on_frontier"]]
    cost_label = cost.replace("_", " ")

    fig = Figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot()
    ax.scatter(
        runs[cost],
        runs[loss],
        s=16,
        color=DOMINATED_COLOR,
        linewidths=0,
        label="Runs",
    )
    ax.scatter(
        front[cost],
        front[loss],
        s=44,
        color=PARETO_COLOR,
        edgecolors="white",
        linewidths=1,
        zorder=3,
        label="Pareto frontier",
    )
    if loss in fit:
        _draw_fit(ax, fit, cost, loss, runs[cost].max(), label)
        targets = fit[fit["target"]]
        for i, (x, y) in enumerate(zip(targets[cost], targets[loss], strict=True)):
            # Above and below in turn, so the labels of close targets do not overlap
            ax.annotate(
                f"{y:.3g}",
                (x, y),
                textcoords="offset points",
                xytext=(8, 6) if i % 2 == 0 else (8, -16),
                color=INK,
                fontweight="bold",
            )
    ax.set(xscale="log", yscale="log")
    _plain_log_labels(ax)
    _style(ax, f"{_label(loss)} over {cost_label}", cost_label.capitalize(), _label(loss))
    return fig


def plot_parameter_over_cost(
    runs: pd.DataFrame, parameter: str, cost: str, fit: pd.DataFrame, label: str
) -> Figure:
    """One parameter of every run over its `cost`, coloured by loss, with the runs on
    the Pareto frontier circled and what the fit `label` predicts for it, if it can.
    """
    front = runs[runs["on_frontier"]]
    cost_label = cost.replace("_", " ")

    fig = Figure(figsize=(7, 4.2))
    ax = fig.add_subplot()
    points = ax.scatter(runs[cost], runs[parameter], c=runs["loss"], s=30, zorder=2)
    ax.scatter(
        front[cost],
        front[parameter],
        s=110,
        facecolors="none",
        edgecolors=PARETO_COLOR,
        linewidths=1.6,
        zorder=3,
        label="Pareto frontier",
    )
    if parameter in fit:
        _draw_fit(ax, fit, cost, parameter, runs[cost].max(), label)
    ax.set(xscale="log", yscale="log")
    _style(ax, f"{parameter} over {cost_label}", cost_label.capitalize(), parameter)
    fig.colorbar(points, ax=ax, label="Loss")
    return fig


def _pareto(
    runs: pd.DataFrame, metric: str, cost: str, configs: Mapping[str, dict]
) -> pd.DataFrame:
    """The runs on the Pareto frontier of `metric` over `cost`, by cost, with their
    model size, data and config.
    """
    front = runs[runs["on_frontier"]].sort_values(cost)
    table = pd.DataFrame(
        {
            "trial_id": front["trial_id"],
            cost: front[cost],
            _name(metric): _own_units(metric, front[metric]),
            "n_params": front["n_params"],
            "n_data": front["n_data"],
        }
    )
    config = pd.DataFrame([configs[i] for i in front["trial_id"]], index=front.index)
    return table.join(config, rsuffix="_config").dropna(axis=1, how="all")


def _predictions(fit: pd.DataFrame, metric: str, cost: str) -> pd.DataFrame:
    """What the fit predicts at the target costs."""
    table = fit[fit["target"]].drop(columns="target")
    table[metric] = _own_units(metric, table[metric])
    return table.rename(columns={cost: f"target_{cost}", metric: _name(metric)})


def _tables(name: str, table: pd.DataFrame) -> list[Artifact]:
    """`table` as a CSV, and aligned in a text file to read."""
    text = table.to_string(index=False, float_format="{:.4g}".format)
    return [
        Artifact(name, table, ArtifactType.CSV),
        Artifact(name, text + "\n", ArtifactType.TEXT),
    ]


def plot_scaling(
    trials: Sequence[Trial],
    predictor: str = "power_law",
    targets: Mapping[str, Sequence[float]] | None = None,
    space: SearchSpace | None = None,
    estimators: Estimators | None = None,
) -> list[Artifact]:
    """The scaling plots and predictions of the loss and of each downstream metric,
    over the cost and, if reported, the inference cost, if any successful run reports
    a cost.

    Over the cost, the fit is `predictor`, the scaling law of the run's optimizer, and
    over the inference cost a power law, each for what it can predict. They are
    extended to `targets`, the target costs by cost, `DEFAULT_TARGETS` times the
    largest cost of the runs by default. The GP posterior needs the cost estimator of
    `estimators`, without it the fit is a power law.
    """
    from scalefm.prediction import Target, get_predictor_type

    table = runs_table(trials)
    if table.empty:
        return []

    downstream = [c for c in table.columns if c.startswith("downstream/")]
    costs = _costs(table)
    parameters = []
    for parameter in table.columns.drop([*COLUMNS, *downstream, *costs[1:]]):
        values = table[parameter]
        if (
            values.isna().any()
            or values.nunique() < 2
            or np.allclose(values, table["cost"])
        ):
            continue
        parameters.append(parameter)

    configs = {trial.id: dict(trial.config) for trial in trials}
    artifacts = []
    for metric in ["loss", *downstream]:
        folder = "pre_train" if metric == "loss" else metric.replace(".", "_")
        for cost in costs:
            runs = _with_frontier(table, cost, metric)
            target_costs = _targets(table, cost, targets)
            law = _law(predictor, cost, estimators)
            provides = [t for t in Target if t in get_predictor_type(law).provides]
            if law == "gp_posterior":
                # Each GP point is its own search over configs, so the curve beyond
                # the runs would be noise, not a trend: draw it over the runs only.
                num, end = NUM_GP_FIT_POINTS, runs[cost].max()
            else:
                num, end = NUM_FIT_POINTS, max(target_costs)
            xs = list(np.geomspace(runs[cost].min(), end, num))
            fit = predict_over(
                runs, metric, cost, law, xs + target_costs, provides, space, estimators
            )
            fit["target"] = [False] * len(xs) + [True] * len(target_costs)
            label = f"{law} fit"

            name = f"{folder}/{_name(metric)}_over_{cost}"
            figure = plot_loss_over_cost(runs, metric, cost, fit, label)
            points = runs[["trial_id", cost, metric, "on_frontier"]]
            artifacts += [
                Artifact(name, figure, ArtifactType.FIGURE),
                Artifact(name, points, ArtifactType.CSV),
            ]
            artifacts += _tables(
                f"{folder}/pareto_over_{cost}", _pareto(runs, metric, cost, configs)
            )
            if metric in fit:
                artifacts += _tables(
                    f"{folder}/predictions_over_{cost}", _predictions(fit, metric, cost)
                )
            for parameter in parameters:
                name = f"{folder}/parameters/{parameter.replace('.', '_')}_over_{cost}"
                figure = plot_parameter_over_cost(runs, parameter, cost, fit, label)
                columns = ["trial_id", cost, parameter, "loss", "on_frontier"]
                artifacts += [
                    Artifact(name, figure, ArtifactType.FIGURE),
                    Artifact(name, runs[columns], ArtifactType.CSV),
                ]
    return artifacts


def _power_of_ten(value: float) -> str:
    """`value` as `10^13` or `3x10^13`, in matplotlib's math text."""
    exponent = int(np.floor(np.log10(value)))
    mantissa = round(value / 10**exponent, 1)
    if mantissa == 1:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:g}\\times10^{{{exponent}}}$"


def _plain_log_labels(ax: Axes) -> None:
    """Label the log y axis with plain numbers, e.g. 2, 5, 10, instead of 2x10^0."""
    low, high = ax.get_ylim()
    shown = list(range(1, 10))
    if high / low > 10:
        shown = [1, 2, 5]
    elif high / low > 3:
        shown = [1, 2, 3, 5]

    def label(value: float, _position: int) -> str:
        mantissa = round(value / 10 ** np.floor(np.log10(value)))
        return f"{value:g}" if mantissa in shown else ""

    ax.yaxis.set_major_formatter(FuncFormatter(label))
    ax.yaxis.set_minor_formatter(FuncFormatter(label))


def _style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Light axes and grid, so the data stands out."""
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=SECONDARY_INK)
    ax.set_ylabel(ylabel, color=SECONDARY_INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS_COLOR)
    ax.tick_params(which="both", color=AXIS_COLOR, labelcolor=SECONDARY_INK)
    ax.grid(visible=True, which="major", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=SECONDARY_INK)


def _target_colors(n_targets: int) -> list[str]:
    """A blue per target budget, darker for larger budgets."""
    if n_targets == 1:
        return [TARGET_COLORS[-2]]
    steps = np.linspace(0, len(TARGET_COLORS) - 1, n_targets).round()
    return [TARGET_COLORS[int(step)] for step in steps]


def _plot_history(history: pd.DataFrame, metric: str, cost: str) -> Figure:
    """The prediction at each target cost over the training cost spent so far."""
    fig = Figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot()
    targets = history[f"target_{cost}"].unique()
    for target, color in zip(targets, _target_colors(len(targets)), strict=True):
        rows = history[history[f"target_{cost}"] == target]
        ax.plot(
            rows["cost_spent"],
            rows[metric],
            color=color,
            linewidth=2,
            drawstyle="steps-post",
            marker="o",
            markersize=6,
            markeredgecolor="white",
            label=f"Target {_power_of_ten(target)}",
        )
    ax.set(xscale="log", yscale="log")
    _plain_log_labels(ax)
    title = f"Prediction convergence, {_label(metric)} over {cost.replace('_', ' ')}"
    ylabel = "Predicted loss" if metric == "loss" else f"Predicted {_label(metric)}"
    _style(ax, title, "Cumulative training cost", ylabel)
    return fig


def plot_convergence(
    trials: Sequence[Trial],
    predictor: str,
    targets: Mapping[str, Sequence[float]] | None = None,
    space: SearchSpace | None = None,
    estimators: Estimators | None = None,
) -> list[Artifact]:
    """For the loss and each downstream metric, over each cost, the prediction at the
    target costs as the runs finished, refitted at most 20 times, to see whether it has
    settled. Saved to `convergence/`, with the fits of `plot_scaling`, for which the
    GP posterior needs `space` and the cost estimator of `estimators`.
    """
    from scalefm.prediction import Target

    finished = sorted(trials, key=lambda t: t.metadata.time_end or float("inf"))
    table = runs_table(finished)
    if table.empty:
        return []

    downstream = [c for c in table.columns if c.startswith("downstream/")]
    artifacts = []
    for metric in ["loss", *downstream]:
        for cost in _costs(table):
            target_costs = _targets(table, cost, targets)
            law = _law(predictor, cost, estimators)
            if law == "gp_posterior" and space is None:
                law = "power_law"
            history = []
            for n in sorted({int(n) for n in np.linspace(3, len(table), 20)}):
                runs = table.iloc[:n].dropna(subset=[metric, cost])
                fit = predict_over(
                    runs,
                    metric,
                    cost,
                    law,
                    target_costs,
                    [Target.LOSS],
                    space,
                    estimators,
                )
                if metric in fit:
                    fit["cost_spent"] = table["cost"].iloc[:n].sum()
                    history.append(fit.rename(columns={cost: f"target_{cost}"}))
            if not history:
                continue

            name = f"convergence/{_name(metric)}_over_{cost}"
            history_table = pd.concat(history, ignore_index=True)
            figure = _plot_history(history_table, metric, cost)
            artifacts += [
                Artifact(name, figure, ArtifactType.FIGURE),
                Artifact(name, history_table, ArtifactType.CSV),
            ]
    return artifacts


def log_to_tensorboard(trials: Sequence[Trial], log_dir: Path) -> None:
    """Log the scaling data to TensorBoard, one step per run, adding only new runs.

    A run's step is the order it was logged in, which is about the order the runs
    finished. At every step:

    - `scaling/incumbent/loss`: the best loss so far
    - `scaling/frontier/...`: the runs on each Pareto frontier so far, as
      `(trial id, value, cost)`

    The ids of the logged runs are kept, in order, in `log_dir/logged_trial_ids.txt`,
    so that later updates, other workers and `scalefm.analyze()` add only the new
    runs, at the next steps.
    It never raises: if logging fails, or another worker holds the lock for longer
    than `TENSORBOARD_LOCK_TIMEOUT`, it warns and the runs are logged at the next
    update.
    """
    finished = sorted(trials, key=lambda t: t.metadata.time_end or float("inf"))
    try:
        table = runs_table(finished)
        if table.empty:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        locker = FileLocker(
            log_dir / ".tensorboard.lock", poll=0.1, timeout=TENSORBOARD_LOCK_TIMEOUT
        )
        with locker.lock():
            _log_new_runs(table, log_dir)
    except pl.exceptions.LockException:
        logger.warning(
            "TensorBoard is busy with another worker, the runs are logged to it at"
            " the next update."
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Could not log the scaling data to TensorBoard, retrying at the next"
            f" update: {e}"
        )


def _log_new_runs(table: pd.DataFrame, log_dir: Path) -> None:
    """Log the runs of `table` that are not logged yet."""
    logged_file = log_dir / "logged_trial_ids.txt"
    logged = logged_file.read_text().split() if logged_file.exists() else []
    is_new = ~table["trial_id"].isin(logged)
    if not is_new.any():
        return

    from torch.utils.tensorboard.writer import SummaryWriter

    first_step = len(logged) + 1
    with tempfile.TemporaryDirectory() as staging:
        writer = SummaryWriter(log_dir=staging)
        downstream = [c for c in table.columns if c.startswith("downstream/")]
        new = table[is_new]
        for i, trial_id in enumerate(new["trial_id"]):
            logged.append(trial_id)
            step = len(logged)
            seen = pd.concat([table[~is_new], new.iloc[: i + 1]])
            writer.add_scalar("scaling/incumbent/loss", seen["loss"].min(), step)
            for cost in _costs(table):
                for loss in ["loss", *downstream]:
                    tag = f"scaling/frontier/{loss}_over_{cost}"
                    writer.add_text(tag, _frontier_text(seen, cost, loss), step)
        writer.close()
        [event_file] = Path(staging).iterdir()
        shutil.move(event_file, log_dir / f"events.out.tfevents.{first_step:010d}")
    logged_file.write_text("\n".join(logged))


def _frontier_text(table: pd.DataFrame, cost: str, loss: str) -> str:
    """The runs on the frontier of `loss` over `cost`, as `(trial id, value, cost)`."""
    front = _with_frontier(table, cost, loss)
    front = front[front["on_frontier"]].sort_values(cost)
    values = front[loss]
    if _is_higher_better(loss.removeprefix("downstream/")):
        values = 1.0 - values
    points = [
        f"({trial_id}, {value:.4g}, {run_cost:.4g})"
        for trial_id, value, run_cost in zip(
            front["trial_id"], values, front[cost], strict=True
        )
    ]
    return ", ".join(points)
