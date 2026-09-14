"""Plots, with the CSV of their points, saved to the summary folder by
`neps.run(..., live_plots=True)`.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure

from neps.optimizers.optimizer import Artifact, ArtifactType

if TYPE_CHECKING:
    from neps.runtime import ResourceUsage
    from neps.state.trial import Trial


def _objectives(trial: Trial) -> list[float]:
    assert trial.report is not None
    score = trial.report.objective_to_minimize
    assert score is not None
    if isinstance(score, Sequence):
        return [float(s) for s in score]
    return [float(score)]


def plot_incumbent_trajectory(
    trials: Sequence[Trial],
    cumulative_usage: Sequence[ResourceUsage],
    incumbent_ids: Collection[str],
) -> list[Artifact]:
    """Plot every evaluated objective and the incumbent over the cumulative cost,
    or over the evaluations if no cost was reported.

    Args:
        trials: The evaluated trials, in chronological order.
        cumulative_usage: The usage of the run up to and including each trial.
        incumbent_ids: The ids of the trials that became the incumbent.

    Returns:
        The figure and a CSV with one row per trial: its cumulative usage, its
        objective, the incumbent's objective so far and whether it became the
        incumbent.
    """
    use_cost = any(u.cost for u in cumulative_usage)
    xs = [u.cost if use_cost else u.evaluations for u in cumulative_usage]
    objectives = [_objectives(t)[0] for t in trials]

    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.scatter(xs, objectives, s=12, alpha=0.4, color="tab:gray", label="Evaluated")
    incumbents = [
        (x, y)
        for t, x, y in zip(trials, xs, objectives, strict=True)
        if t.id in incumbent_ids
    ]
    incumbent_xs = [x for x, _ in incumbents] + [xs[-1]]
    incumbent_ys = [y for _, y in incumbents]
    ax.step(
        incumbent_xs,
        [*incumbent_ys, incumbent_ys[-1]],
        where="post",
        color="tab:blue",
        label="Incumbent",
    )
    ax.set(
        xlabel="Cumulative cost" if use_cost else "Evaluations",
        ylabel="Objective to minimize",
        title="Incumbent trajectory",
    )
    ax.legend()

    rows: list[dict[str, Any]] = []
    best = float("inf")
    for trial, usage, objective in zip(trials, cumulative_usage, objectives, strict=True):
        best = min(best, objective)
        rows.append(
            {
                "trial_id": trial.id,
                **usage.to_trajectory_dict(),
                "objective_to_minimize": objective,
                "incumbent_objective_to_minimize": best,
                "is_incumbent": trial.id in incumbent_ids,
            }
        )

    return [
        Artifact("incumbent_trajectory", fig, ArtifactType.FIGURE),
        Artifact("incumbent_trajectory", rows, ArtifactType.CSV),
    ]


def plot_pareto_front(
    trials: Sequence[Trial],
    pareto_ids: Collection[str],
) -> list[Artifact]:
    """Plot two objectives against each other highlighting the Pareto front
    Args:
        trials: The evaluated trials.
        pareto_ids: The ids of the trials on the Pareto front.

    Returns:
        The figure and a CSV with one row per trial: its two objectives and whether
        it is on the Pareto front.
    """
    objectives = [_objectives(t) for t in trials]

    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.scatter(
        [o[0] for o in objectives],
        [o[1] for o in objectives],
        s=12,
        alpha=0.4,
        color="tab:gray",
        label="Evaluated",
    )
    front = sorted(
        o for t, o in zip(trials, objectives, strict=True) if t.id in pareto_ids
    )
    ax.step(
        [o[0] for o in front],
        [o[1] for o in front],
        where="post",
        marker="o",
        color="tab:red",
        label="Pareto front",
    )
    ax.set(xlabel="Objective 0", ylabel="Objective 1", title="Pareto front")
    ax.legend()

    rows = [
        {
            "trial_id": t.id,
            "objective_0": o[0],
            "objective_1": o[1],
            "on_pareto_front": t.id in pareto_ids,
        }
        for t, o in zip(trials, objectives, strict=True)
    ]

    return [
        Artifact("pareto_front", fig, ArtifactType.FIGURE),
        Artifact("pareto_front", rows, ArtifactType.CSV),
    ]
