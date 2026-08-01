"""Utility functions for reading and processing results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import neps
from neps.state.trial import State


def process_seed(
    *,
    path: str | Path,
    seed: str | int | None,
    key_to_extract: str | None = None,
) -> tuple[list[float], list[float], float]:
    """Reads and processes data per seed."""
    path = Path(path)
    if seed is not None:
        path = path / f"seed_{seed}"

    _fulldf, _summary = neps.status(path, print_summary=False)
    if _fulldf.empty:
        raise ValueError(f"No trials found in {path}")

    # _fulldf = _fulldf.sort_values("time_sampled")

    def get_cost(idx: str | int) -> float:
        row = _fulldf.loc[idx]
        if key_to_extract and key_to_extract in row:
            return float(row[key_to_extract])
        return 1.0

    losses = []
    costs = []

    # max_cost only relevant for scaling x-axis when using fidelity on the x-axis
    max_cost: float = -1.0
    global_start = _fulldf["time_sampled"].min()

    for config_id, config_result in _fulldf.iterrows():
        if config_result["state"] != State.SUCCESS:
            continue

        cost = get_cost(config_id)

        loss = float(config_result["objective_to_minimize"])
        losses.append(loss)
        costs.append(cost)
    
    cumsum_costs = list(np.cumsum(costs))
    max_cost = cumsum_costs[-1] if cumsum_costs else -1.0
    
    print(f"Processed seed {seed} for {path}, accumulated cost: {max_cost:.2e}")
    return list(np.minimum.accumulate(losses)), cumsum_costs, max_cost


def process_optimizer(
    *,
    base_path: str | Path,
    optimizer_name: str,
    seeds: list[int | str],
    key_to_extract: str | None = None,
    log_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process all seeds for an optimizer, interpolate to common flop counts, and average.
    
    Args:
        base_path: Base path where optimizer directories are located
        optimizer_name: Name of the optimizer directory
        seeds: List of seed values to process
        key_to_extract: Key to extract from results as cost (typically 'flopcount')
        consider_continuations: Whether to consider continuation costs for MF algorithms
        n_workers: Number of workers used
        log_scale: Whether to interpolate on log scale for flop counts
    
    Returns:
        flop_counts: Common flop count grid (x-axis)
        mean_losses: Averaged losses across seeds
        std_losses: Standard deviation of losses across seeds
    """
    base_path = Path(base_path)
    optimizer_path = base_path / optimizer_name
    
    all_seeds_data = []
    
    # Process each seed
    for seed in seeds:
        try:
            losses, costs, max_cost = process_seed(
                path=optimizer_path,
                seed=seed,
                key_to_extract=key_to_extract,
            )
            all_seeds_data.append({"losses": losses, "costs": costs, "max_cost": max_cost})
        except Exception as e:
            print(f"Warning: Could not process {optimizer_name} seed {seed}: {e}")
            continue
    
    if not all_seeds_data:
        raise ValueError(f"No valid seeds found for optimizer {optimizer_name}")
    
    # Create common grid for interpolation
    max_cost_overall = max(d["max_cost"] for d in all_seeds_data)
    
    if log_scale:
        # Log scale interpolation
        flop_grid = np.logspace(
            np.log10(min(c[0] for d in all_seeds_data for c in [d["costs"]] if c)),
            np.log10(max_cost_overall),
            num=100
        )
    else:
        # Linear scale interpolation
        flop_grid = np.linspace(0, max_cost_overall, num=100)
    
    interpolated_losses = []
    
    for seed_data in all_seeds_data:
        costs = np.array(seed_data["costs"])
        losses = np.array(seed_data["losses"])
        
        # Interpolate to common grid
        interp_losses = np.interp(flop_grid, costs, losses, left=losses[0], right=losses[-1])
        interpolated_losses.append(interp_losses)
    
    interpolated_losses_array = np.array(interpolated_losses)
    
    # Calculate mean and std
    mean_losses = np.mean(interpolated_losses_array, axis=0)
    std_losses = np.std(interpolated_losses_array, axis=0)
    
    return flop_grid, mean_losses, std_losses


def plot_optimizer_rankings(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int | str]],
    key_to_extract: str | None = None,
    log_scale: bool = False,
    figsize: tuple[int, int] = (12, 8),
    title: str = "Optimizer Rankings by Loss vs FLOPs",
    plot_individually: bool = False,
):
    """
    Plot relative rankings of optimizers based on loss with flopcount on x-axis.
    
    Args:
        base_path: Base path where optimizer directories are located
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost (typically 'flopcount')
        log_scale: Whether to use log scale for x-axis
        figsize: Figure size
        title: Plot title
        plot_individually: If True, create separate plots for each optimizer. If False, create comparison plots.
    
    Returns:
        Figure(s) and dictionary with optimizer results for further analysis
    """
    import matplotlib.pyplot as plt
    
    results = {}
    
    # Process all optimizers first
    colors = plt.cm.tab20(np.linspace(0, 1, len(optimizers)))
    
    for idx, (opt_name, seeds) in enumerate(optimizers.items()):
        flop_counts, mean_losses, std_losses = process_optimizer(
            base_path=base_path,
            optimizer_name=opt_name,
            seeds=seeds,
            key_to_extract=key_to_extract,
            log_scale=log_scale,
        )
        
        results[opt_name] = {
            "flop_counts": flop_counts,
            "mean_losses": mean_losses,
            "std_losses": std_losses,
        }
    
    if plot_individually:
        # Create individual plots for each optimizer
        figs = {}
        for idx, (opt_name, data) in enumerate(results.items()):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            color = colors[idx]
            
            flop_counts = data["flop_counts"]
            mean_losses = data["mean_losses"]
            std_losses = data["std_losses"]
            
            # Plot 1: Loss vs FLOPs with confidence bands
            ax1.plot(
                flop_counts,
                mean_losses,
                color=color,
                linewidth=2.5,
                label=opt_name,
            )
            ax1.fill_between(
                flop_counts,
                mean_losses - std_losses,
                mean_losses + std_losses,
                alpha=0.2,
                color=color,
            )
            
            ax1.set_xlabel("FLOPs", fontsize=12)
            ax1.set_ylabel("Loss", fontsize=12)
            ax1.set_title(f"{opt_name}: Loss vs FLOPs", fontsize=14, fontweight="bold")
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            if log_scale:
                ax1.set_xscale("log")
            
            # Plot 2: Relative ranking over optimization
            losses_at_flop_all = [results[o]["mean_losses"] for o in results.keys()]
            rankings = []
            for i in range(len(flop_counts)):
                losses_at_flop = [losses[i] for losses in losses_at_flop_all]
                rank = sorted(losses_at_flop).index(mean_losses[i]) + 1
                rankings.append(rank)
            
            ax2.plot(
                flop_counts,
                rankings,
                color=color,
                linewidth=2.5,
                marker="o",
                markersize=6,
                label=opt_name,
            )
            
            ax2.set_xlabel("FLOPs", fontsize=12)
            ax2.set_ylabel("Relative Rank (1=best)", fontsize=12)
            ax2.set_title(f"{opt_name}: Ranking Over Optimization", fontsize=14, fontweight="bold")
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()  # Invert y-axis so rank 1 is at top
            if log_scale:
                ax2.set_xscale("log")
            
            plt.tight_layout()
            figs[opt_name] = fig
        
        return figs, results
    else:
        # Create comparison plots with all optimizers
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        for idx, (opt_name, data) in enumerate(results.items()):
            color = colors[idx]
            flop_counts = data["flop_counts"]
            mean_losses = data["mean_losses"]
            std_losses = data["std_losses"]
            
            # Plot 1: Loss vs FLOPs with confidence bands
            ax1.plot(
                flop_counts,
                mean_losses,
                label=opt_name,
                color=color,
                linewidth=2.5,
            )
            ax1.fill_between(
                flop_counts,
                mean_losses - std_losses,
                mean_losses + std_losses,
                alpha=0.2,
                color=color,
            )
        
        ax1.set_xlabel("FLOPs", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        if log_scale:
            ax1.set_xscale("log")
        
        # Plot 2: Relative ranking over optimization
        for opt_name, data in results.items():
            flop_counts = data["flop_counts"]
            mean_losses = data["mean_losses"]
            
            opt_idx = list(optimizers.keys()).index(opt_name)
            color = colors[opt_idx]
            
            # Calculate ranking at each flop count
            rankings = []
            for i, flop in enumerate(flop_counts):
                flop_idx = i
                losses_at_flop = [
                    results[o]["mean_losses"][flop_idx]
                    for o in results.keys()
                ]
                rank = sorted(losses_at_flop).index(mean_losses[i]) + 1
                rankings.append(rank)
            
            ax2.plot(
                flop_counts,
                rankings,
                label=opt_name,
                color=color,
                linewidth=2.5,
                marker="o",
                markersize=4,
                alpha=0.7,
            )
        
        ax2.set_xlabel("FLOPs", fontsize=12)
        ax2.set_ylabel("Relative Rank (1=best)", fontsize=12)
        ax2.set_title("Optimizer Rankings Over Optimization", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Invert y-axis so rank 1 is at top
        if log_scale:
            ax2.set_xscale("log")
        
        plt.tight_layout()
        
        return fig, results
