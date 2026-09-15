"""Aggregates the per-worker-count sweeps into a throughput-vs-workers table and figure.
Run after the jobs submitted by `run_scaling_study.py` have finished.
"""

from __future__ import annotations

import pandas as pd
from run_scaling_study import (
    ROOT_DIRECTORY,
    TOTAL_EVALUATIONS,
    WORKER_COUNTS,
    root_dir_for,
)

import neps

SUMMARY_DIR = ROOT_DIRECTORY / "summary"

_COLUMNS = {
    "config.lr": "lr",
    "config.wd": "wd",
    "config.batch_size": "batch_size",
    "extra.wall_clock_time_sec": "wall_clock_time_sec",
    "extra.samples_per_sec": "samples_per_sec",
    "extra.total_train_samples": "total_train_samples",
}


def _sweep_df(n_workers: int) -> pd.DataFrame:
    root_dir = root_dir_for(n_workers)
    if not root_dir.exists():
        return pd.DataFrame()

    df, _ = neps.status(root_directory=root_dir)
    if df.empty or "extra.samples_per_sec" not in df.columns:
        return pd.DataFrame()

    df = df.rename(columns=_COLUMNS).dropna(subset=["samples_per_sec"])
    df["n_workers"] = n_workers
    keep = [
        "n_workers",
        "lr",
        "wd",
        "batch_size",
        "wall_clock_time_sec",
        "total_train_samples",
        "samples_per_sec",
        "time_started",
        "time_end",
    ]
    return df[[c for c in keep if c in df.columns]]


def _all_trials() -> pd.DataFrame:
    frames = [f for f in (_sweep_df(n) for n in WORKER_COUNTS) if not f.empty]
    if not frames:
        raise RuntimeError(
            f"No finished trials found under {ROOT_DIRECTORY}. Run "
            "run_scaling_study.py and wait for its Slurm jobs to finish first."
        )
    return pd.concat(frames, ignore_index=True)


def _sweep_row(group: pd.DataFrame) -> pd.Series:
    sweep_sec = group["time_end"].max() - group["time_started"].min()
    return pd.Series(
        {
            "n_trials": len(group),
            "sweep_wall_clock_sec": sweep_sec,
            "sweep_samples_per_sec": group["total_train_samples"].sum() / sweep_sec,
            "per_trial_samples_per_sec": group["samples_per_sec"].median(),
            "per_trial_samples_per_sec_min": group["samples_per_sec"].min(),
            "per_trial_samples_per_sec_max": group["samples_per_sec"].max(),
        }
    )


def performance_report() -> pd.DataFrame:
    trials = _all_trials()
    table = (
        trials.groupby("n_workers")
        .apply(_sweep_row, include_groups=False)
        .reset_index()
        .sort_values("n_workers")
        .reset_index(drop=True)
    )
    table["n_trials"] = table["n_trials"].astype(int)
    table["speedup"] = (
        table["sweep_samples_per_sec"] / table.loc[0, "sweep_samples_per_sec"]
    )

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(SUMMARY_DIR / "scaling_table.csv", index=False)

    _plot_scaling(table)
    return table


def _plot_scaling(table: pd.DataFrame) -> None:
    """Sweep throughput against the number of workers, on log-log axes so that
    linear scaling reads as a straight line of slope 1.
    """
    import matplotlib.pyplot as plt

    workers = table["n_workers"].to_numpy()
    sweep = table["sweep_samples_per_sec"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(workers, sweep, "o-", color="tab:blue", linewidth=1.8, markersize=6)
    for x, y in zip(workers, sweep, strict=False):
        ax.annotate(
            f"{y:,.0f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8.5,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_ylim(sweep.min() / 1.25, sweep.max() * 1.45)
    ax.set_xticks(workers)
    ax.set_xticklabels(workers)
    ax.set_xlabel("Parallel workers (one GPU each)")
    ax.set_ylabel("Training throughput (training samples per second)")
    ax.set_title(
        f"Throughput scaling of a fixed {TOTAL_EVALUATIONS}-evaluation "
        "hyperparameter optimization",
        fontsize=11,
    )
    ax.grid(visible=True, which="both", alpha=0.3, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(SUMMARY_DIR / "scaling_study.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    performance_report()
