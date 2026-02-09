import abc
from logging import getLogger
from typing import Callable, TYPE_CHECKING, Mapping, Any, Sequence
from neps.optimizers.optimizer import AskFunction, Artifact, ArtifactType

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from neps.space import SearchSpace
from neps.state import BudgetInfo, Trial
from neps.optimizers.optimizer import SampledConfig
if TYPE_CHECKING:
    pass

logger = getLogger(__name__)
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

    @abc.abstractmethod
    def adapt_search_space(self, trials: Mapping[str, Trial], max_evaluation_flops: int) -> None:
        """Tailor the pipeline based on scaling laws."""
        pass

    FLOPS_KEY = "flops"
    N_PARAM_KEY = "n_param"
    N_DATA_KEY = "n_data"
    

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
                n_params = trial.report.extra[cls.N_PARAM_KEY] * 1_000_000
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
                n_params = trial.report.extra[cls.N_PARAM_KEY] * 1_000_000
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
    def _plot_pareto_front(cls, trials: Mapping[str, Trial]) -> plt.Figure | None:
        """Generate Pareto front visualization (non-dominated points) in log-log scale."""
        from matplotlib.colors import Normalize
        
        rows = []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                continue
            obj = trial.report.objective_to_minimize
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY] * 1_000_000
                n_data = trial.report.extra[cls.N_DATA_KEY]
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Could not extract metrics for trial {trial.id}, skipping plot point. {e}")
                continue
            rows.append((n_params, n_data, float(obj), flops,))

        if not rows:
            logger.warning("No evaluated trials with objectives/flops to plot.")
            return None

        pareto_front = cls._compute_pareto_front(rows)

        if not pareto_front:
            logger.warning("No Pareto front points found.")
            return None

        pareto_front.sort(key=lambda r: r[3])
        
        n_params_front = [r[0] for r in pareto_front]
        n_data_front = [r[1] for r in pareto_front]
        objs_front = [r[2] for r in pareto_front]
        flops_front = [r[3] for r in pareto_front]
        
        n_params_all = [r[0] for r in rows]
        objs_all = [r[2] for r in rows]
        flops_all = [r[3] for r in rows]
        
        log_flops_front = np.log(np.array(flops_front))
        log_objs_front = np.log(np.array(objs_front))
        
        n_points = len(log_flops_front)
        weights = np.exp(np.linspace(0, 4, n_points))
        weights = weights / np.sum(weights)
        
        try:
            if not np.all(np.isfinite(log_flops_front)) or not np.all(np.isfinite(log_objs_front)):
                logger.warning("Non-finite values in log-transformed data. Using unweighted fit.")
                weights = None
            if weights is not None and not np.all(np.isfinite(weights)):
                logger.warning("Weights contain non-finite values. Using unweighted fit.")
                weights = None
            
            coeffs = np.polyfit(log_flops_front, log_objs_front, 1, w=weights)
            slope, intercept = coeffs[0], coeffs[1]
            
            if not (np.isfinite(slope) and np.isfinite(intercept)):
                logger.warning(f"Fitted parameters are non-finite: slope={slope}, intercept={intercept}")
                slope, intercept, fitted_objs = None, None, None
            else:
                fit_line = np.poly1d(coeffs)
                fitted_log_objs = fit_line(log_flops_front)
                fitted_objs = np.exp(fitted_log_objs)
        except Exception as e:
            logger.warning(f"Could not fit linear trend to Pareto front: {e}")
            slope, intercept, fitted_objs = None, None, None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(flops_all, objs_all, c=n_params_all, cmap='gray', marker='o', 
                  alpha=0.2, s=30, label='All points')
        
        norm_front = Normalize(vmin=min(n_params_front), vmax=max(n_params_front))
        sc = ax.scatter(flops_front, objs_front, c=n_params_front, cmap='inferno', 
                       marker='D', s=100, alpha=1.0, edgecolors='black', linewidth=1.5,
                       norm=norm_front, label='Pareto front (actual)')
        
        ax.plot(flops_front, objs_front, 'k-', alpha=0.3, linewidth=1)
        
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
                n_params = trial.report.extra[cls.N_PARAM_KEY] * 1_000_000
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
                n_params = trial.report.extra[cls.N_PARAM_KEY] * 1_000_000
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
                fig_pareto = cls._plot_pareto_front(trials)
                if fig_pareto is not None:
                    artifacts.append(
                        Artifact("pareto_front", fig_pareto, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate Pareto front plot: {e}")
            
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
                fig_envelope = cls._plot_training_curve_envelope(trials)
                if fig_envelope is not None:
                    artifacts.append(
                        Artifact("training_curve_envelope", fig_envelope, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate training curve envelope plot: {e}")
        
        return artifacts
