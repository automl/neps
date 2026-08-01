import abc
from logging import getLogger
from typing import Callable, TYPE_CHECKING, Mapping, Any, Sequence
from neps.optimizers.optimizer import AskFunction, Artifact, ArtifactType
import math

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from neps.space import SearchSpace
from neps.state import BudgetInfo, Trial
from neps.optimizers.optimizer import SampledConfig

logger = getLogger(__name__)


def _extract_trial_data(
    trials: Mapping[str, Trial],
    lc_aware: bool = False,
) -> tuple[list[tuple], dict]:
    """Extract and normalize trial data from trials mapping.
    
    Args:
        trials: Mapping of trial IDs to Trial objects.
        lc_aware: If True, extract learning curves; otherwise extract final metrics.
    
    Returns:
        Tuple of (rows, trial_id_map) where:
            - rows: List of tuples (n_params, n_data, loss, flops)
            - trial_id_map: Dict mapping row tuples to trial IDs
    """
    FLOPS_KEY = "flops"
    N_PARAM_KEY = "n_param"
    N_DATA_KEY = "n_data"
    
    rows = []
    trial_id_map = {}
    
    for trial in trials.values():
        if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
            continue
        if trial.report.extra is None:
            continue
        
        try:
            flops = trial.report.extra[FLOPS_KEY]
            n_params = trial.report.extra[N_PARAM_KEY]
            n_data = trial.report.extra[N_DATA_KEY]
        except KeyError as e:
            logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
            continue
        except Exception as e:
            logger.error(f"Could not extract metrics for trial {trial.id}, skipping. {e}")
            continue
        
        obj = float(trial.report.objective_to_minimize)
        
        if lc_aware and trial.report.learning_curve is not None and len(trial.report.learning_curve) > 0:
            lc = trial.report.learning_curve
            if isinstance(lc, (list, tuple)) and len(lc) > 0:
                for tokens_val, flops_val, loss_val in lc:
                    row = (n_params, tokens_val, loss_val, flops_val)
                    rows.append(row)
                    trial_id_map[row] = trial.id
        else:
            row = (n_params, n_data, obj, flops)
            rows.append(row)
            trial_id_map[row] = trial.id
    
    return rows, trial_id_map


def _fit_loglog_line(
    flops_values: list[float],
    loss_values: list[float],
) -> tuple[float | None, float | None, list[float] | None]:
    """Fit a line in log-log space with weighting.
    
    Args:
        flops_values: List of FLOP counts.
        loss_values: List of corresponding loss values.
    
    Returns:
        Tuple of (slope, intercept, fitted_objs) or (None, None, None) if fitting failed.
    """
    if not flops_values or not loss_values:
        return None, None, None
    
    log_flops = np.log(np.array(flops_values))
    log_objs = np.log(np.array(loss_values))
    
    # Compute weights (emphasize higher FLOPS)
    # try:
    #     if not np.all(np.isfinite(log_flops)) or not np.all(np.isfinite(log_objs)):
    #         logger.warning("Non-finite values in log-transformed data. Using unweighted fit.")
    #         weights = None
    #     else:
    #         max_log_flops, min_log_flops = np.max(log_flops), np.min(log_flops)
    #         if max_log_flops - min_log_flops < 1e-8:
    #             logger.warning("FLOPS values are too close. Using unweighted fit.")
    #             weights = None
    #         else:
    #             weights = np.exp(3 * (log_flops - min_log_flops) / (max_log_flops - min_log_flops + 1e-8))
    #             if weights is not None and not np.all(np.isfinite(weights)):
    #                 logger.warning("Weights contain non-finite values. Using unweighted fit.")
    #                 weights = None
    # except Exception as e:
    #     logger.warning(f"Could not compute weights: {e}")
    weights = None
    
    # Fit linear regression in log-log space
    try:
        coeffs = np.polyfit(log_flops, log_objs, 1, w=weights)
        slope, intercept = coeffs[0], coeffs[1]
        
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            logger.warning(f"Fitted parameters are non-finite: slope={slope}, intercept={intercept}")
            return None, None, None
        
        fit_line = np.poly1d(coeffs)
        fitted_log_objs = fit_line(log_flops)
        fitted_objs = np.exp(fitted_log_objs)
        
        return slope, intercept, fitted_objs.tolist()
    except Exception as e:
        logger.warning(f"Could not fit linear trend: {e}")
        return None, None, None


def fit_pareto_front_loglog(
    trials: Mapping[str, Trial],
    lc_aware: bool = False,
) -> tuple[float | None, float | None, np.ndarray | None, list[float], list[float], list[tuple], list[tuple], set]:
    """Fit a line to the Pareto front in log-log space.
    
    Extracts the Pareto front from trials, enforces monotonicity, and fits a linear
    regression in log-log space: log(loss) = slope * log(flops) + intercept
    
    Args:
        trials: Mapping of trial IDs to Trial objects.
        lc_aware: If True, extract and smooth learning curves before computing Pareto front.
    
    Returns:
        Tuple of (slope, intercept, fitted_objs, flops_front, objs_front, pareto_front, all_rows, pareto_trial_ids) where:
            - slope: Scaling exponent (None if fitting failed)
            - intercept: Log-space intercept (None if fitting failed)
            - fitted_objs: Predicted losses at pareto front flops (None if fitting failed)
            - flops_front: FLOPs values of pareto front
            - objs_front: Loss values of pareto front
            - pareto_front: List of pareto front tuples (n_params, n_data, loss, flops)
            - all_rows: List of all valid trial tuples (n_params, n_data, loss, flops)
            - pareto_trial_ids: Set of trial IDs that are on the Pareto front
    """
    # Extract trial data
    rows, trial_id_map = _extract_trial_data(trials, lc_aware=lc_aware)
    
    if not rows:
        logger.warning("No evaluated trials to fit Pareto front.")
        return None, None, None, [], [], [], [], set()
    
    # Compute Pareto front
    pareto = []
    for i, (n_i, d_i, l_i, f_i) in enumerate(rows):
        dominated = False
        for j, (_, _, l_j, f_j) in enumerate(rows):
            if i == j:
                continue
            if (f_j <= f_i and l_j <= l_i) and (f_j < f_i or l_j < l_i):
                dominated = True
                break
        if not dominated:
            pareto.append((n_i, d_i, l_i, f_i))
    
    if not pareto:
        logger.warning("No Pareto front points found.")
        return None, None, None, [], [], [], rows, set()
    
    # Enforce monotonicity
    if len(pareto) > 1:
        sorted_front = sorted(pareto, key=lambda x: x[3], reverse=True)
        monotonic_front = [sorted_front[0]]
        prev_n = sorted_front[0][0]
        
        for i in range(1, len(sorted_front)):
            n_p, n_d, loss, flops = sorted_front[i]
            if n_p <= prev_n:
                monotonic_front.append((n_p, n_d, loss, flops))
                prev_n = n_p
            else:
                logger.debug(f"Removing non-monotonic point: flops={flops:.2e}, loss={loss:.4f}")
        
        pareto = monotonic_front
    
    if not pareto:
        logger.warning("No Pareto front points found after enforcing monotonicity.")
        return None, None, None, [], [], [], rows, set()
    
    # Collect trial IDs on Pareto front
    pareto_trial_ids = {trial_id_map.get(row) for row in pareto if trial_id_map.get(row) is not None}
    
    # Sort by FLOPs and extract data
    pareto.sort(key=lambda r: r[3])
    objs_front = [r[2] for r in pareto]
    flops_front = [r[3] for r in pareto]
    
    # Fit line in log-log space
    slope, intercept, fitted_objs = _fit_loglog_line(flops_front, objs_front)
    
    return slope, intercept, fitted_objs, flops_front, objs_front, pareto, rows, pareto_trial_ids


def fit_convex_hull_loglog(
    trials: Mapping[str, Trial],
    lc_aware: bool = False,
) -> tuple[float | None, float | None, np.ndarray | None, list[float], list[float], list[tuple], list[tuple], set]:
    """Fit a line to the convex hull in log-log space.
    
    Extracts the convex hull from trials in (flops, loss) space, enforces monotonicity,
    and fits a linear regression in log-log space: log(loss) = slope * log(flops) + intercept
    
    Args:
        trials: Mapping of trial IDs to Trial objects.
        lc_aware: If True, extract and smooth learning curves before computing convex hull.
    
    Returns:
        Tuple of (slope, intercept, fitted_objs, flops_hull, objs_hull, convex_hull, all_rows, hull_trial_ids) where:
            - slope: Scaling exponent (None if fitting failed)
            - intercept: Log-space intercept (None if fitting failed)
            - fitted_objs: Predicted losses at convex hull flops (None if fitting failed)
            - flops_hull: FLOPs values of convex hull
            - objs_hull: Loss values of convex hull
            - convex_hull: List of convex hull tuples (n_params, n_data, loss, flops)
            - all_rows: List of all valid trial tuples (n_params, n_data, loss, flops)
            - hull_trial_ids: Set of trial IDs that are on the convex hull
    """
    from scipy.spatial import ConvexHull
    
    # Extract trial data
    rows, trial_id_map = _extract_trial_data(trials, lc_aware=lc_aware)
    
    if not rows:
        logger.warning("No evaluated trials to fit convex hull.")
        return None, None, None, [], [], [], [], set()
    
    # Prepare points for convex hull in (log_flops, log_loss) space
    points_for_hull = []
    row_indices = []
    
    for i, (n_p, n_d, loss, flops) in enumerate(rows):
        if loss > 0 and flops > 0:
            try:
                log_flops = np.log(flops)
                log_loss = np.log(loss)
                if np.isfinite(log_flops) and np.isfinite(log_loss):
                    points_for_hull.append([log_flops, log_loss])
                    row_indices.append(i)
            except Exception as e:
                logger.debug(f"Could not log-transform point (flops={flops}, loss={loss}): {e}")
    
    if len(points_for_hull) < 3:
        logger.warning(f"Not enough valid points ({len(points_for_hull)}) for convex hull computation (need >= 3).")
        return None, None, None, [], [], [], rows, set()
    
    try:
        hull = ConvexHull(points_for_hull)
        hull_indices = hull.vertices
    except Exception as e:
        logger.warning(f"Could not compute convex hull: {e}")
        return None, None, None, [], [], [], rows, set()
    
    # Extract convex hull points and filter to lower hull only
    hull_points_with_indices = []
    
    for idx in hull_indices:
        original_row_idx = row_indices[idx]
        hull_points_with_indices.append((rows[original_row_idx], original_row_idx))
    
    if not hull_points_with_indices:
        logger.warning("No convex hull points found.")
        return None, None, None, [], [], [], rows, set()
    
    # Sort by FLOPs
    hull_points_with_indices.sort(key=lambda x: x[0][3])
    
    # Extract lower hull: keep only points where loss monotonically decreases
    lower_hull = []
    min_loss = float('inf')
    
    for row, _ in hull_points_with_indices:
        if row[2] < min_loss:  # row[2] is loss
            lower_hull.append(row)
            min_loss = row[2]
    
    if not lower_hull:
        logger.warning("No lower hull points found.")
        return None, None, None, [], [], [], rows, set()
    
    # Map back to trial IDs
    convex_hull = lower_hull
    hull_trial_ids = set()
    for row in convex_hull:
        trial_id = trial_id_map.get(row)
        if trial_id is not None:
            hull_trial_ids.add(trial_id)
    
    # Extract data
    objs_hull = [r[2] for r in convex_hull]
    flops_hull = [r[3] for r in convex_hull]
    
    # Fit line in log-log space
    slope, intercept, fitted_objs = _fit_loglog_line(flops_hull, objs_hull)
    
    return slope, intercept, fitted_objs, flops_hull, objs_hull, convex_hull, rows, hull_trial_ids


@dataclass
class ScalingLawGuidedOptimizer:
    space: SearchSpace
    base_optimizer: AskFunction
    max_evaluation_flops: float
    max_target_flops: float
    flops_estimator: Callable
    metric_functions: Mapping[str, Callable]
        

    def __call__(
            self, trials: Mapping[str, Trial], 
            budget_info: BudgetInfo | None = None, n: int | None = None,
        ) -> SampledConfig | list[SampledConfig]:
        return self.base_optimizer(
            trials=trials,
            budget_info=budget_info,
            n=n,
        )

    @abc.abstractmethod
    def extrapolate(self, trials: Mapping[str, Trial], max_target_flops: int) -> tuple[dict[str, Any], float]:
        """Extrapolate the performance of a trial to the target flops."""
        pass

    # @abc.abstractmethod
    def adapt_search_space(self, trials: Mapping[str, Trial], max_evaluation_flops: int) -> None:
        """Tailor the pipeline based on scaling laws."""
        pass

    FLOPS_KEY = "flops"
    N_PARAM_KEY = "n_param"
    N_DATA_KEY = "n_data"
    LEARNING_CURVE_KEY = "learning_curve"
    

    @classmethod
    def _plot_training_curve_envelope(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate training curves showing loss vs FLOPs for different parameter counts (no disk I/O)."""
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                logger.warning(f"Trial {trial.id} has no extra data. Skipping.")
                continue
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY]
                obj = float(trial.report.objective_to_minimize)
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}: {e}")
                continue
            if n_params <= 0 or flops <= 0:
                continue
            rows.append((flops, obj, n_params))

        if len(rows) < 2:
            logger.warning("Not enough evaluated trials to plot training curve envelope.")
            return None

        rows.sort(key=lambda r: r[2])
        
        param_counts = {}
        for flops, obj, n_params in rows:
            param_bin = round(np.log10(n_params) * 4) / 4
            if param_bin not in param_counts:
                param_counts[param_bin] = []
            param_counts[param_bin].append((flops, obj))

        if not param_counts:
            logger.warning("No valid parameter groups found.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        
        param_levels = sorted(param_counts.keys())
        cmap = plt.cm.get_cmap('viridis')
        norm = plt.Normalize(vmin=min(param_levels), vmax=max(param_levels))
        
        for param_bin in param_levels:
            points = param_counts[param_bin]
            points.sort(key=lambda p: p[0])
            
            if len(points) >= 2:
                flops_vals = [p[0] for p in points]
                obj_vals = [p[1] for p in points]
                
                param_count = 10 ** param_bin
                color = cmap(norm(param_bin))
                
                ax.plot(flops_vals, obj_vals, marker='o', linestyle='-', 
                       color=color, alpha=0.7, linewidth=2, markersize=4,
                       label=f"{param_count:.2e}" if len(param_levels) <= 10 else "")

        ax.set_xscale('log')
        ax.set_xlabel('FLOPs', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Curve Envelope: Loss vs FLOPs (colored by Model Size)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Log10(Parameters)', fontsize=11)
        
        plt.tight_layout()
        return fig
    

    @classmethod
    def extract_pareto_front_data(cls, trials: Mapping[str, Trial], lc_aware: bool, pareto_fit: bool) -> dict[str, list]:
        fit_func = fit_pareto_front_loglog if pareto_fit else fit_convex_hull_loglog
        result = fit_func(trials, lc_aware=lc_aware)
        pareto_rows = result[5]
        
        return {
            'flops': [r[3] for r in pareto_rows],
            'params': [float(r[0]) for r in pareto_rows],
            'data': [float(r[1]) for r in pareto_rows],
            'loss': [r[2] for r in pareto_rows],
        }
    
    @classmethod
    def _plot_flops_per_objective(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate FLOPs vs objective visualization (no disk I/O)."""
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY]
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}, skipping plot point. {e}")
                continue
            rows.append((n_params, obj, flops))

        if not rows:
            logger.warning("No evaluated trials with objectives/flops to plot.")
            return None

        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplementedError("Multi-objective Scaling law not implemented yet.")

        n_param = [r[0] for r in rows]
        xs = [r[2] for r in rows]
        ys = [float(r[1]) for r in rows]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=min(n_param), vmax=max(n_param))
        
        sc = ax.scatter(xs, ys, c=n_param, cmap='inferno', marker='o', alpha=0.9, norm=norm)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs FLOPs (log-log scale)")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="Parameters")
        return fig

    @classmethod
    def _compute_pareto_front(cls, trial_data: list[tuple]) -> list[tuple]:
        """Compute Pareto front from trial data (n, d, loss, flops)."""
        pareto = []
        for i, (n_i, d_i, l_i, f_i) in enumerate(trial_data):
            dominated = False
            for j, (_, _, l_j, f_j) in enumerate(trial_data):
                if i == j:
                    continue
                if (f_j <= f_i and l_j <= l_i) and (f_j < f_i or l_j < l_i):
                    dominated = True
                    break
            if not dominated:
                pareto.append((n_i, d_i, l_i, f_i))
        return pareto

    @classmethod
    def _plot_pareto_front(cls, trials: Mapping[str, Trial]) -> tuple[plt.Figure | None, str | None]:
        """Generate Pareto front visualization (non-dominated points) in log-log scale.
        
        Returns:
            Tuple of (figure, csv_string) where csv_string contains Pareto front trial configs.
        """
        from matplotlib.colors import Normalize
        
        # Get Pareto front and fit using the standalone function
        slope, intercept, fitted_objs, flops_front, objs_front, pareto_front, rows, _ = fit_pareto_front_loglog(trials)
        
        if not pareto_front:
            logger.warning("No Pareto front points found.")
            return None, None
        
        # Build trial_map for CSV generation
        trial_map = {}
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY]
                n_data = trial.report.extra[cls.N_DATA_KEY]
            except (KeyError, Exception):
                continue
            row = (n_params, n_data, float(obj), flops)
            trial_map[row] = trial
        
        # Extract data from pareto front
        n_params_front = [r[0] for r in pareto_front]
        n_data_front = [r[1] for r in pareto_front]
        
        n_params_all = [r[0] for r in rows]
        objs_all = [r[2] for r in rows]
        flops_all = [r[3] for r in rows]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(flops_all, objs_all, c=n_params_all, cmap='gray', marker='o', 
                  alpha=0.2, s=30, label='All points')
        
        norm_front = Normalize(vmin=min(n_params_front), vmax=max(n_params_front))
        sc = ax.scatter(flops_front, objs_front, c=n_params_front, cmap='inferno', 
                       marker='D', s=100, alpha=1.0, edgecolors='black', linewidth=1.5,
                       norm=norm_front, label='Pareto front (actual)')
        
        if fitted_objs is not None and slope is not None:
            ax.plot(flops_front, fitted_objs, 'r--', alpha=0.7, linewidth=2.5, 
                   label=f'Linear fit (log scale, slope={slope:.4f})')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel("FLOPs", fontsize=12)
        ax.set_ylabel("Objective to minimize", fontsize=12)
        ax.set_title(f"Pareto Front - Log-Log Scale ({len(pareto_front)}/{len(rows)} points)", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.3, which='both')
        ax.legend(loc='best')
        
        if slope is not None and intercept is not None:
            scaler = np.exp(intercept)
            line_eq = f"Log-linear fit: log(y) = {slope:.4f}·log(x) + {intercept:.4f}  →  y = {scaler:.4e}·x^{slope:.4f}"
            fig.text(0.5, 0.02, line_eq, ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.colorbar(sc, ax=ax, label="Parameters")
        
        # Generate CSV with Pareto front trial configs
        try:
            import csv
            import io
            import json
            
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            
            # Write header
            header = ["trial_id", "n_params", "n_data", "objective", "flops", "config"]
            writer.writerow(header)
            
            # Write Pareto front trials
            for row in pareto_front:
                n_p, n_d, obj, flops = row
                trial = trial_map.get(row)
                trial_id = trial.id if trial else "unknown"
                config_str = json.dumps(dict(trial.config)) if trial and trial.config else "{}"
                writer.writerow([trial_id, n_p, n_d, obj, flops, config_str])
            
            csv_string = csv_buffer.getvalue()
        except Exception as e:
            logger.warning(f"Failed to generate Pareto front CSV: {e}")
            csv_string = None
        
        return fig, csv_string

    @classmethod
    def compute_fit_weights(cls, log_flops_front, log_objs_front):
        if not np.all(np.isfinite(log_flops_front)) or not np.all(np.isfinite(log_objs_front)):
            logger.warning("Non-finite values in log-transformed data. Using unweighted fit.")
            return None
        max_log_flops, min_log_flops = np.max(log_flops_front), np.min(log_flops_front)
        if max_log_flops - min_log_flops < 1e-8:
            logger.warning("FLOPs values for Pareto front are too close. Using unweighted fit.")
            return None
        weights = np.exp(3 * (log_flops_front - min_log_flops) / (max_log_flops - min_log_flops + 1e-8))
        if weights is not None and not np.all(np.isfinite(weights)):
            logger.warning("Weights contain non-finite values. Using unweighted fit.")
            return None
        return weights

    @classmethod
    def _plot_pareto_front_params_vs_data(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate Pareto front visualization (non-dominated points) for params vs data points in log-log scale."""
        from matplotlib.colors import Normalize
        
        slope, intercept, fitted_objs, flops_front, objs_front, pareto_front, rows, _ = fit_pareto_front_loglog(trials)
        
        if not pareto_front:
            logger.warning("No Pareto front points found for params vs data.")
            return None

        pareto_front.sort(key=lambda r: r[1])  # Sort by data points
        
        n_params_front = [r[0] for r in pareto_front]
        n_data_front = [r[1] for r in pareto_front]
        objs_front = [r[2] for r in pareto_front]
        
        n_params_all = [r[0] for r in rows]
        n_data_all = [r[1] for r in rows]
        objs_all = [r[2] for r in rows]
        
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(n_data_all, n_params_all, c=objs_all, cmap='gray', marker='o', 
                  alpha=0.2, s=30, label='All points')
        
        norm_front = Normalize(vmin=min(objs_front), vmax=max(objs_front))
        sc = ax.scatter(n_data_front, n_params_front, c=objs_front, cmap='inferno', 
                       marker='D', s=100, alpha=1.0, edgecolors='black', linewidth=1.5,
                       norm=norm_front, label='Pareto front (actual)')
        
        ax.plot(n_data_front, n_params_front, 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel("Data Points", fontsize=12)
        ax.set_ylabel("Parameters", fontsize=12)
        ax.set_title(f"Pareto Front (Params vs Data) - Log-Log Scale ({len(pareto_front)}/{len(rows)} points)", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.3, which='both')
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="Objective (Loss)")
        return fig

    @classmethod
    def _plot_params_vs_loss(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate params vs loss visualization colored by time (no disk I/O)."""
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                n_params = trial.report.extra[cls.N_PARAM_KEY]
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}, skipping plot point. {e}")
                continue
            rows.append((trial.metadata.time_sampled, obj, n_params))

        if not rows:
            logger.warning("No evaluated trials with objectives/params to plot.")
            return None

        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplementedError("Multi-objective Scaling law not implemented yet.")

        times = [r[0] for r in rows]
        xs = [r[2] for r in rows]
        ys = [float(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(xs, ys, c=times, cmap='coolwarm', marker='o', alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs Parameters (colored by Time)")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="Time")
        return fig

    @classmethod
    def _plot_data_vs_loss(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate data points vs loss visualization colored by FLOPs (no disk I/O)."""
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                n_data = trial.report.extra[cls.N_DATA_KEY]
                flops = trial.report.extra[cls.FLOPS_KEY]
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}, skipping plot point. {e}")
                continue
            rows.append((n_data, float(obj), flops))

        if not rows:
            logger.warning("No evaluated trials with data/loss to plot.")
            return None

        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplementedError("Multi-objective Scaling law not implemented yet.")

        xs = [r[0] for r in rows]
        ys = [float(r[1]) for r in rows]
        flops = [r[2] for r in rows]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=min(flops), vmax=max(flops))
        
        sc = ax.scatter(xs, ys, c=flops, cmap='plasma', marker='o', alpha=0.9, norm=norm)
        ax.set_xlabel("Number of Data Points")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs Data Points")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.colorbar(sc, ax=ax, label="FLOPs")
        return fig

    @classmethod
    def _plot_accumulated_flops_per_objective(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate accumulated FLOPs vs objective visualization (no disk I/O)."""
        rows = []
        accumulated_flops = 0
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}, skipping plot point. {e}")
                continue
            accumulated_flops += flops
            rows.append((trial.id, obj, accumulated_flops))

        if not rows:
            logger.warning("No evaluated trials with objectives/flops to plot.")
            return None

        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplementedError("Multi-objective Scaling law not implemented yet.")

        xs = [r[2] for r in rows]
        ys = [float(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker='o', linestyle='-', alpha=0.9)
        ax.set_xlabel("Accumulated FLOPs")
        ax.set_ylabel("Objective to minimize")
        ax.set_title("Objective vs Accumulated FLOPs")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        return fig

    @classmethod
    def _plot_flop_vs_param(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate FLOPs vs Params visualization (no disk I/O)."""
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                continue
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY]
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}: {e}")
                continue
            if n_params == 0:
                continue
            rows.append((flops, n_params))

        if not rows:
            logger.warning("No evaluated trials with flops/params to plot.")
            return None

        xs = [r[0] for r in rows]
        ys = [float(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(xs, ys, marker='o', alpha=0.9)
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Params")
        ax.set_title("FLOPs vs Params")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        return fig

    @classmethod
    def _normalize_search_loss_flops(
        cls,
        losses: list[float],
        accumulated_flops_list: list[float],
        global_loss_bounds: tuple[float, float] | None = None,
        global_flops_bounds: tuple[float, float] | None = None,
        log_loss_scale: bool = False,
    ) -> tuple[list[float], list[float], list[float], tuple[float, float], tuple[float, float]] | None:
        """Filter and normalize losses and per-trial FLOPs to [0, 1]."""
        if not losses or not accumulated_flops_list or len(losses) != len(accumulated_flops_list):
            logger.warning("Invalid input: Lists must be populated and of equal length.")
            return None

        valid_pairs = []
        for loss, accumulated_flops in zip(losses, accumulated_flops_list):
            try:
                loss_value = float(loss)
                accumulated_flops_value = float(accumulated_flops)
            except (TypeError, ValueError):
                continue

            if (
                math.isfinite(loss_value)
                and math.isfinite(accumulated_flops_value)
                and accumulated_flops_value >= 0.0
            ):
                valid_pairs.append((loss_value, accumulated_flops_value))
        if not valid_pairs:
            logger.warning("Invalid input: No finite loss/FLOP pairs available.")
            return None

        losses = [loss for loss, _accumulated_flops in valid_pairs]
        accumulated_flops_list = [accumulated_flops for _loss, accumulated_flops in valid_pairs]
            
        # 1. Derive Actual FLOPs (The physical cost of each model)
        actual_flops = [accumulated_flops_list[0]]
        for i in range(1, len(accumulated_flops_list)):
            cost = accumulated_flops_list[i] - accumulated_flops_list[i-1]
            actual_flops.append(cost)
            
        # 2. Log-transform FLOPs and optionally losses.
        # Use max(1.0, x) to prevent math domain errors if a step accidentally recorded 0.
        if log_loss_scale:
            transformed_losses = [math.log10(max(1.0, loss)) for loss in losses]
        else:
            transformed_losses = losses
        log_flops = [math.log10(max(1.0, f)) for f in actual_flops]
        # log_flops = actual_flops
        
        # 3. Establish Global Bounds for the Unit Square
        # If not provided, dynamically infer from the run (less ideal for multi-algo comparisons)
        min_loss = global_loss_bounds[0] if global_loss_bounds else min(transformed_losses)
        max_loss = global_loss_bounds[1] if global_loss_bounds else max(transformed_losses)
        
        min_log_flops = global_flops_bounds[0] if global_flops_bounds else min(log_flops)
        max_log_flops = global_flops_bounds[1] if global_flops_bounds else max(log_flops)
        
        # Prevent divide-by-zero if bounds are identical
        loss_range = max(1e-9, max_loss - min_loss)
        flops_range = max(1e-9, max_log_flops - min_log_flops)
        
        # 4. Normalize everything to [0, 1]
        norm_losses = [(l - min_loss) / loss_range for l in transformed_losses]
        norm_flops = [(f - min_log_flops) / flops_range for f in log_flops]

        return (
            accumulated_flops_list,
            norm_losses,
            norm_flops,
            (min_loss, max_loss),
            (min_log_flops, max_log_flops),
        )

    @staticmethod
    def _non_dominated_indices(
        *,
        losses: Sequence[float],
        flops: Sequence[float],
    ) -> list[int]:
        """Return Pareto-optimal indices for minimized (flops, loss)."""
        pareto_indices: list[int] = []
        for j in range(len(losses)):
            dominated = False
            for k in range(len(losses)):
                if j == k:
                    continue
                if (
                    flops[k] <= flops[j]
                    and losses[k] <= losses[j]
                    and (flops[k] < flops[j] or losses[k] < losses[j])
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(j)
        return pareto_indices

    @staticmethod
    def _fit_normalized_pareto_line(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
        """Fit y = slope*x + intercept in normalized Pareto space."""
        finite_points = [
            (float(x), float(y))
            for x, y in points
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        if len(finite_points) < 2:
            return None

        xs = np.asarray([p[0] for p in finite_points], dtype=float)
        ys = np.asarray([p[1] for p in finite_points], dtype=float)
        if len(np.unique(xs)) < 2:
            return None

        try:
            slope, intercept = np.polyfit(xs, ys, 1)
        except Exception as e:
            logger.warning(f"Could not fit normalized Pareto line: {e}")
            return None
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return None
        return float(slope), float(intercept)

    @staticmethod
    def _area_between_normalized_lines(
        *,
        line_a: tuple[float, float],
        line_b: tuple[float, float],
        x_left: float,
        x_right: float,
    ) -> float:
        """Integrate absolute vertical distance between two normalized lines."""
        if x_right <= x_left:
            return 0.0

        slope_diff = float(line_a[0] - line_b[0])
        intercept_diff = float(line_a[1] - line_b[1])

        candidates = [float(x_left), float(x_right)]
        if abs(slope_diff) > 1e-12:
            root = -intercept_diff / slope_diff
            if x_left < root < x_right:
                candidates.append(float(root))
        candidates = sorted(candidates)

        area = 0.0
        for start, stop in zip(candidates[:-1], candidates[1:]):
            mid = 0.5 * (start + stop)
            sign = 1.0 if (slope_diff * mid + intercept_diff) >= 0.0 else -1.0
            integral = 0.5 * slope_diff * (stop**2 - start**2) + intercept_diff * (stop - start)
            area += sign * integral
        return float(max(0.0, area))

    @classmethod
    def compute_normalized_search_hypervolume(
        cls,
        losses: list[float], 
        accumulated_flops_list: list[float],
        global_loss_bounds: tuple[float, float] | None = None,
        global_flops_bounds: tuple[float, float] | None = None
    ) -> tuple[list[float], list[float]] | None:
        """
        Computes the normalized Hypervolume (search efficiency) over accumulated FLOPs.
        Maps physical costs to a [0,1] unit square to ensure fair area calculations.
        
        Args:
            losses: List of loss values in chronological order of evaluation.
            accumulated_flops_list: List of total cluster FLOPs spent up to that step.
            global_loss_bounds: (min_possible_loss, max_starting_loss). 
            global_flops_bounds: (min_log10_flops, max_log10_flops) of your search grid.
            
        Returns:
            Tuple of (X_axis: accumulated_flops, Y_axis: hypervolume_over_time)
        """
        normalized = cls._normalize_search_loss_flops(
            losses=losses,
            accumulated_flops_list=accumulated_flops_list,
            global_loss_bounds=global_loss_bounds,
            global_flops_bounds=global_flops_bounds,
            log_loss_scale=False,
        )
        if normalized is None:
            return None

        accumulated_flops_list, norm_losses, norm_flops, _loss_bounds, _flops_bounds = normalized
        
        # We fix the reference point outside the unit square to ensure the worst
        # possible valid model still generates a non-zero hypervolume.
        ref_point = (1.1, 1.1)
        max_hypervolume = ref_point[0] * ref_point[1]
        
        hypervolumes = []
        
        # 5. Calculate Hypervolume progression over the search timeline
        for i in range(len(norm_losses)):
            objs_so_far = norm_losses[:i+1]
            flops_so_far = norm_flops[:i+1]
            
            # --- Pareto Filter (Using NORMALIZED Physical Costs) ---
            pareto_indices = cls._non_dominated_indices(losses=objs_so_far, flops=flops_so_far)
                    
            # Sort left-to-right by normalized FLOPs
            pareto_points = [(flops_so_far[idx], objs_so_far[idx]) for idx in pareto_indices]
            pareto_points.sort(key=lambda p: p[0])
            
            # --- Sum the Rectangles ---
            hv = 0.0

            for point_idx, (flops_p, obj_p) in enumerate(pareto_points):
                if point_idx + 1 < len(pareto_points):
                    next_flops = pareto_points[point_idx + 1][0]
                else:
                    next_flops = ref_point[0]

                width = next_flops - flops_p
                height = ref_point[1] - obj_p
                hv += width * height

            hv = min(hv, max_hypervolume)
            if hypervolumes:
                hv = max(hv, hypervolumes[-1])
            hypervolumes.append(hv)
            
        return accumulated_flops_list, hypervolumes

    @classmethod
    def compute_normalized_pareto_line_area_to_true(
        cls,
        losses: list[float],
        accumulated_flops_list: list[float],
        true_pareto_info: Mapping[str, Sequence[float]],
        global_loss_bounds: tuple[float, float] | None = None,
        global_flops_bounds: tuple[float, float] | None = None,
        log_loss_scale: bool = True,
    ) -> tuple[list[float], list[float]] | None:
        """
        Compute area between fitted current Pareto line and fitted true-Pareto line.

        Works directly in log10(loss) vs log10(per-trial-flops) space with no
        min/max normalization, so rankings are stable regardless of which other
        optimizers are included in the comparison. Lower is better.

        global_loss_bounds and global_flops_bounds are accepted for backward
        compatibility but are intentionally ignored.
        """
        _ = (global_loss_bounds, global_flops_bounds)

        if true_pareto_info is None:
            logger.warning("true_pareto_info is required.")
            return None

        true_flops_raw = np.asarray(true_pareto_info.get("flops", []), dtype=float)
        true_losses_raw = np.asarray(true_pareto_info.get("loss", []), dtype=float)
        true_mask = (
            np.isfinite(true_flops_raw)
            & np.isfinite(true_losses_raw)
            & (true_flops_raw > 0)
            & (true_losses_raw > 0)
        )
        if np.sum(true_mask) < 2:
            logger.warning("Need at least two finite true-Pareto points.")
            return None

        # Validate and filter input pairs
        valid_pairs = []
        for loss, acc_flops in zip(losses, accumulated_flops_list):
            try:
                lv, fv = float(loss), float(acc_flops)
            except (TypeError, ValueError):
                continue
            if math.isfinite(lv) and math.isfinite(fv) and fv >= 0.0:
                valid_pairs.append((lv, fv))
        if not valid_pairs:
            logger.warning("Invalid input: No finite loss/FLOP pairs available.")
            return None

        clean_losses = [p[0] for p in valid_pairs]
        clean_acc_flops = [p[1] for p in valid_pairs]

        # Derive per-trial FLOPs from cumulative totals
        actual_flops = [clean_acc_flops[0]]
        for i in range(1, len(clean_acc_flops)):
            actual_flops.append(clean_acc_flops[i] - clean_acc_flops[i - 1])

        # Log-transform only — no min/max scaling so rankings don't depend on
        # which other optimizers happen to be in the comparison set.
        if log_loss_scale:
            log_losses = [math.log10(max(1e-12, l)) for l in clean_losses]
        else:
            log_losses = list(clean_losses)
        log_flops = [math.log10(max(1.0, f)) for f in actual_flops]

        # True Pareto in the same log space
        if log_loss_scale:
            true_log_losses = np.log10(np.maximum(1e-12, true_losses_raw[true_mask]))
        else:
            true_log_losses = true_losses_raw[true_mask].astype(float)
        true_log_flops = np.log10(np.maximum(1.0, true_flops_raw[true_mask]))

        true_pareto_indices = cls._non_dominated_indices(
            losses=true_log_losses.tolist(),
            flops=true_log_flops.tolist(),
        )
        true_pareto_points = [
            (float(true_log_flops[idx]), float(true_log_losses[idx]))
            for idx in true_pareto_indices
        ]
        true_pareto_points.sort(key=lambda p: p[0])
        true_line = cls._fit_normalized_pareto_line(true_pareto_points)
        if true_line is None:
            logger.warning("Could not fit true-Pareto line.")
            return None

        x_left = float(min(p[0] for p in true_pareto_points))
        x_right = float(max(p[0] for p in true_pareto_points))
        if x_right <= x_left:
            logger.warning("True-Pareto FLOP range is empty.")
            return None

        line_areas: list[float] = []
        for i in range(len(log_losses)):
            objs_so_far = log_losses[:i + 1]
            flops_so_far = log_flops[:i + 1]
            pareto_indices = cls._non_dominated_indices(losses=objs_so_far, flops=flops_so_far)
            pareto_points = [(flops_so_far[idx], objs_so_far[idx]) for idx in pareto_indices]
            pareto_points.sort(key=lambda p: p[0])

            current_line = cls._fit_normalized_pareto_line(pareto_points)
            if current_line is None:
                line_areas.append(float("nan"))
                continue

            area = cls._area_between_normalized_lines(
                line_a=current_line,
                line_b=true_line,
                x_left=x_left,
                x_right=x_right,
            )
            line_areas.append(area)

        return clean_acc_flops, line_areas

    @classmethod
    def compute_search_pareto_front(
        cls,
        losses: list[float],
        flops_list: list[float],
        global_loss_bounds: tuple[float, float] | None = None,
        global_flops_bounds: tuple[float, float] | None = None,
    ) -> tuple[list[float], list[float], list[int]] | None:
        """
        Computes non-dominated Pareto-front points over accumulated FLOPs.

        Dominance is computed directly in raw objective space using FLOPs and loss
        (both minimized), without normalization. The global bounds arguments are
        accepted for backward compatibility but intentionally ignored.

        Args:
            losses: List of loss values in chronological order of evaluation.
            accumulated_flops_list: List of total cluster FLOPs spent up to that step.
            global_loss_bounds: Unused (kept for backward compatibility).
            global_flops_bounds: Unused (kept for backward compatibility).

        Returns:
            Tuple of (pareto_flops, pareto_losses, pareto_indices) sorted by FLOPs, 
            where pareto_indices are the original indexes from flops_list that are Pareto optimal,
            or None on invalid input.
        """
        if not losses or not flops_list or len(losses) != len(flops_list):
            logger.warning("Invalid input: Lists must be populated and of equal length.")
            return None

        flops_arr = np.asarray(flops_list, dtype=float)
        losses_arr = np.asarray(losses, dtype=float)

        
        valid_mask = np.isfinite(flops_arr) & np.isfinite(losses_arr)
        if not np.any(valid_mask):
            logger.warning("No finite FLOPs/loss values available to compute Pareto front.")
            return None

        flops_valid = flops_arr[valid_mask]
        losses_valid = losses_arr[valid_mask]
        valid_indices = np.where(valid_mask)[0]  # Track original indices in input lists
        
        # Keep signature compatibility with existing callers while explicitly
        # using raw-space dominance for Pareto filtering.
        _ = (global_loss_bounds, global_flops_bounds)

        pareto_indices_in_valid: list[int] = []
        for i in range(len(losses_valid)):
            dominated = False
            for j in range(len(losses_valid)):
                if i == j:
                    continue
                if (
                    flops_valid[j] <= flops_valid[i]
                    and losses_valid[j] <= losses_valid[i]
                    and (flops_valid[j] < flops_valid[i] or losses_valid[j] < losses_valid[i])
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_indices_in_valid.append(i)

        if not pareto_indices_in_valid:
            logger.warning("No Pareto front points found.")
            return None

        pareto_flops = flops_valid[pareto_indices_in_valid]
        pareto_losses = losses_valid[pareto_indices_in_valid]
        
        # Map back to original indices in input lists
        pareto_indices_in_input = [valid_indices[i] for i in pareto_indices_in_valid]

        order = np.argsort(pareto_flops)
        return pareto_flops[order].tolist(), pareto_losses[order].tolist(), [pareto_indices_in_input[i] for i in order]

    @classmethod
    def _plot_hypervolume_over_time(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate hypervolume (area under Pareto front) over accumulated FLOPs visualization."""
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            obj = trial.report.objective_to_minimize
            try:
                time_sampled = trial.metadata.time_sampled
                flops = trial.report.extra[cls.FLOPS_KEY]
            except (AttributeError, KeyError) as e:
                logger.warning(f"Trial {trial.id} missing time_sampled or FLOPs: {e}")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}: {e}")
                continue
            rows.append((time_sampled, float(obj), flops))

        if not rows:
            logger.warning("No evaluated trials with time_sampled/FLOPs to plot.")
            return None

        # Sort by time
        rows.sort(key=lambda r: r[0])
        
        # Compute cumulative FLOPs
        accumulated_flops_list = []
        total_flops = 0
        for _, _, flops in rows:
            total_flops += flops
            accumulated_flops_list.append(total_flops)
        
        # Extract losses
        losses = [float(obj) for _, obj, _ in rows]
        
        # Compute hypervolume
        hypervolume_result = cls.compute_normalized_search_hypervolume(losses, accumulated_flops_list)
        
        if hypervolume_result is None:
            logger.warning("Could not compute hypervolume over accumulated FLOPs.")
            return None

        _, hypervolumes = hypervolume_result
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot hypervolume growth
        ax.plot(accumulated_flops_list, hypervolumes, marker='o', linestyle='-', alpha=0.9, 
               linewidth=2.5, color='blue', label='Hypervolume', markersize=6)
        ax.fill_between(accumulated_flops_list, hypervolumes, alpha=0.3, color='blue')
        
        ax.set_xscale('log')
        ax.set_xlabel("Accumulated FLOPs", fontsize=12)
        ax.set_ylabel("Hypervolume (Area)", fontsize=12)
        ax.set_title("Hypervolume (Area Under Pareto Front) Over Accumulated FLOPs", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig

    @classmethod
    def get_trial_artifacts(cls, trials: Mapping[str, Trial] | None = None) -> list[Artifact] | None:
        """Return scaling law artifacts for runtime persistence.

        Consolidates all artifacts: scaling law visualization plots from trials data.

        Args:
            trials: Mapping of trial IDs to Trial objects. Required to generate plots.

        Returns:
            List of Artifact objects, or None if no trials provided.
        """
        artifacts = []
        
        if trials is not None:            
            try:
                fig_accumulated = cls._plot_accumulated_flops_per_objective(trials)
                if fig_accumulated is not None:
                    artifacts.append(
                        Artifact("accumulated_flops_objective", fig_accumulated, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate accumulated FLOPs plot: {e}")
            
            try:
                fig_flops_params = cls._plot_flop_vs_param(trials)
                if fig_flops_params is not None:
                    artifacts.append(
                        Artifact("flops_vs_params", fig_flops_params, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate FLOPs vs Params plot: {e}")
            
            try:
                fig_flops_obj = cls._plot_flops_per_objective(trials)
                if fig_flops_obj is not None:
                    artifacts.append(
                        Artifact("flops_per_objective", fig_flops_obj, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate FLOPs per objective plot: {e}")
            
            try:
                fig_pareto, csv_pareto = cls._plot_pareto_front(trials)
                if fig_pareto is not None:
                    artifacts.append(
                        Artifact("pareto_front", fig_pareto, ArtifactType.FIGURE)
                    )
                if csv_pareto is not None:
                    artifacts.append(
                        Artifact("pareto_front_configs", csv_pareto, ArtifactType.CSV)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate Pareto front plot: {e}")
            
            try:
                fig_pareto_params_data = cls._plot_pareto_front_params_vs_data(trials)
                if fig_pareto_params_data is not None:
                    artifacts.append(
                        Artifact("pareto_front_params_vs_data", fig_pareto_params_data, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate Pareto front (params vs data) plot: {e}")
            
            try:
                fig_params_loss = cls._plot_params_vs_loss(trials)
                if fig_params_loss is not None:
                    artifacts.append(
                        Artifact("params_vs_loss", fig_params_loss, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate params vs loss plot: {e}")
            
            try:
                fig_data_loss = cls._plot_data_vs_loss(trials)
                if fig_data_loss is not None:
                    artifacts.append(
                        Artifact("data_vs_loss", fig_data_loss, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate data vs loss plot: {e}")
            
            try:
                fig_hypervolume = cls._plot_hypervolume_over_time(trials)
                if fig_hypervolume is not None:
                    artifacts.append(
                        Artifact("hypervolume_over_time", fig_hypervolume, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate hypervolume over time plot: {e}")
            
            try:
                fig_envelope = cls._plot_training_curve_envelope(trials)
                if fig_envelope is not None:
                    artifacts.append(
                        Artifact("training_curve_envelope", fig_envelope, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate training curve envelope plot: {e}")
        
        return artifacts
