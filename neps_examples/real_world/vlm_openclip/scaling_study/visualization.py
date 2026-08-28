"""Visualizations built from the three per-GPU-count NePS sweeps.

Run once the jobs submitted by `run_scaling_study.py` have finished:

    python visualization.py
"""

import pandas as pd

import neps
from run_scaling_study import MAIN_LOAD_N_GPUS, N_GPUS_CHOICES, ROOT_DIRECTORY, root_dir_for

SUMMARY_DIR = ROOT_DIRECTORY / "summary"


def _sweep_df(n_gpus: int) -> pd.DataFrame:
    """Every finished trial of one GPU count's sweep, one row each."""
    root_dir = root_dir_for(n_gpus)
    if not root_dir.exists():
        return pd.DataFrame()

    df, _ = neps.status(root_directory=root_dir)
    if df.empty or "extra.samples_per_sec" not in df.columns:
        return pd.DataFrame()

    df = df.rename(columns={
        "config.batch_size": "batch_size",
        "config.lr": "lr",
        "config.wd": "wd",
        "extra.wall_clock_time_sec": "wall_clock_time_sec",
        "extra.samples_per_sec": "samples_per_sec",
        "extra.total_train_samples": "total_train_samples",
    })
    df = df.dropna(subset=["samples_per_sec"])
    df["n_gpus"] = n_gpus
    keep = ["n_gpus", "lr", "wd", "batch_size", "wall_clock_time_sec",
            "total_train_samples", "samples_per_sec"]
    return df[[c for c in keep if c in df.columns]]


def _all_trials() -> pd.DataFrame:
    frames = [_sweep_df(n) for n in N_GPUS_CHOICES]
    trials = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame()

    if trials.empty:
        raise RuntimeError(
            f"No finished trials found under {ROOT_DIRECTORY}. Run "
            "run_scaling_study.py and wait for its Slurm jobs to finish first."
        )
    return trials


def performance_report() -> pd.DataFrame:
    """Aggregate the three sweeps into one throughput-vs-GPUs table and figure.

    Each GPU count ran the same grid of configs, so its sweep gives several
    throughput measurements rather than one. They are aggregated by **median**,
    which is what the figure's line and the `speedup` column are built from --
    a single trial could be a straggler (a slow node, a cold filesystem), and
    the median is not moved by one.

    The spread across a sweep is itself worth seeing, so `samples_per_sec_min`
    and `_max` are kept in the table and every individual trial is drawn on the
    figure behind the line.
    """
    trials = _all_trials()

    table = (
        trials.groupby("n_gpus")
        .agg(
            n_trials=("samples_per_sec", "size"),
            wall_clock_time_sec=("wall_clock_time_sec", "median"),
            samples_per_sec=("samples_per_sec", "median"),
            samples_per_sec_min=("samples_per_sec", "min"),
            samples_per_sec_max=("samples_per_sec", "max"),
        )
        .reset_index()
        .sort_values("n_gpus")
        .reset_index(drop=True)
    )
    table["speedup"] = table["samples_per_sec"] / table.loc[0, "samples_per_sec"]

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    trials.to_csv(SUMMARY_DIR / "scaling_trials.csv", index=False)
    table.to_csv(SUMMARY_DIR / "scaling_table.csv", index=False)
    print(table.to_string(index=False))

    _plot_scaling(table, trials)

    print(f"\nWrote table + plot to {SUMMARY_DIR}")
    return table


def _plot_scaling(table: pd.DataFrame, trials: pd.DataFrame) -> None:
    """Total throughput vs. #GPUs: every trial as a point, the median as the
    line, against the ideal linear reference through the smallest GPU count.
    The gap between the two curves is the cost of parallelising, in the units
    the proposal actually cares about.
    """
    import matplotlib.pyplot as plt

    gpus = table["n_gpus"].to_numpy()
    median = table["samples_per_sec"].to_numpy()
    ideal = median[0] * (gpus / gpus[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    ax.plot(gpus, ideal, "--", color="grey", label="ideal (linear)", zorder=1)
    ax.scatter(
        trials["n_gpus"], trials["samples_per_sec"],
        color="tab:blue", alpha=0.45, s=28, zorder=2,
        label=f"individual trials (n={len(trials)})",
    )
    ax.plot(gpus, median, "o-", color="tab:blue", label="median", zorder=3)
    for x, y in zip(gpus, median):
        ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)

    if MAIN_LOAD_N_GPUS in set(gpus):
        ax.axvline(MAIN_LOAD_N_GPUS, color="tab:red", ls=":",
                   label=f"production load ({MAIN_LOAD_N_GPUS} GPUs)", zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(gpus)
    ax.set_xticklabels(gpus)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Total throughput (samples/sec)")
    ax.set_title("Same job, more GPUs — total throughput")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(SUMMARY_DIR / "scaling_study.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    performance_report()
