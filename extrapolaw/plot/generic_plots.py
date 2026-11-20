"""Plots, with the CSV of their points, saved to the summary folder by
`extrapolaw.acquire(..., live_plots=True)`.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

from matplotlib.figure import Figure

from extrapolaw.optimizers.optimizer import Artifact, ArtifactType

if TYPE_CHECKING:
    from extrapolaw.state.trial import Trial


def _objectives(trial: Trial) -> list[float]:
    assert trial.report is not None
    score = trial.report.objective_to_minimize
    assert score is not None
    if isinstance(score, Sequence):
        return [float(s) for s in score]
    return [float(score)]


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
