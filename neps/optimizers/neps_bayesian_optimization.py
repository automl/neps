from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Callable

import numpy as np
import torch

from neps.optimizers.models.neps_gp import _greedy_batch_acquisition
from neps.optimizers.models.neps_gp import _log_expected_improvement
from neps.optimizers.models.neps_gp import _expected_improvement
from neps.optimizers.models.neps_gp import KernelSurrogateModel
from neps.optimizers.models.neps_gp import _hamming_kernel
from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.optimizers.utils.util import _get_max_trial_id
from neps.space.neps_spaces.parameters import PipelineSpace
from neps.space.neps_spaces.sampling import RandomSampler
from neps.space.neps_spaces.neps_space import _prepare_sampled_configs, resolve

if TYPE_CHECKING:
    from neps.state import BudgetInfo, Trial
    from neps.state.pipeline_eval import UserResultDict




def _extract_training_data(
    trials: Mapping[str, Trial],
) -> tuple[list[dict], np.ndarray]:
    """
    Extract hierarchical configs and objective values from completed trials.

    Args:
        trials: Mapping of trial_id -> Trial objects

    Returns:
        (configs, objectives) where:
        - configs: List of hierarchical config dicts
        - objectives: Array of objective values to minimize
    """
    configs = []
    objectives = []

    for trial in trials.values():
        if trial.report is not None and trial.report.objective_to_minimize is not None:
            # trial.config should contain the hierarchical sampling decisions
            configs.append(trial.config)
            objectives.append(trial.report.objective_to_minimize)

    return configs, np.array(objectives)


@dataclass
class NepsBayesianOptimization:
    """Uses `botorch` as an engine for doing bayesian optimiziation."""

    def __init__(
        self,
        space: PipelineSpace,
        n_initial_design: int = 5,
        device: torch.device | None = None,
        acqu_sampling_density: int = 1000,
        acqu_function: Literal["EI", "LogEI"] | Callable = "EI",
        use_batch_acquisition: bool = False,
        kernel_function: Literal["hamming"] | Callable = "hamming",
    ) -> None:
        self._pipeline = space
        self.n_initial_design = n_initial_design
        self.device = device
        self._random_sampler = RandomSampler(predefined_samplings={})
        self.acqu_sampling_density = acqu_sampling_density
        self.use_batch_acquisition = use_batch_acquisition
        self.acqu_function = acqu_function
        self.kernel = kernel_function

        if isinstance(acqu_function, str):
            match acqu_function:
                case "EI":
                    self.acqu_function = _expected_improvement
                case "LogEI":
                    self.acqu_function = _log_expected_improvement
                case _:
                    raise ValueError(f"Unsupported acquisition function: {acqu_function}")

        if isinstance(kernel_function, str):
            match kernel_function:
                case "hamming":
                    self.kernel = _hamming_kernel
                case _:
                    raise ValueError(f"Unsupported kernel: {kernel_function}")


    def __call__(  # noqa: C901, PLR0912, PLR0915  # noqa: C901, PLR0912
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        """
        Bayesian Optimization for Hierarchical NePS Spaces.
        """

        n_to_sample = 1 if n is None else n
        max_trial_id = _get_max_trial_id(trials)

        # If the amount of configs evaluated is less than the initial design
        # requirement, keep drawing from initial design
        n_evaluated = sum(
            1
            for trial in trials.values()
            if trial.report is not None and trial.report.objective_to_minimize is not None
        )
        sampled_configs: list[SampledConfig] = []
        if n_evaluated < self.n_initial_design:
            chosen_pipelines = [
                resolve(
                    pipeline=self._pipeline,
                    domain_sampler=self._random_sampler,
                    environment_values={},
                )
                for _ in range(n_to_sample)
                ]
            return _prepare_sampled_configs(chosen_pipelines, max_trial_id, n_to_sample==1)

        training_configs, training_y = _extract_training_data(trials)

        if not training_configs:
            raise ValueError("No training data available despite passing initial design phase")

        # STEP 2: Fit kernel-based surrogate model
        surrogate = KernelSurrogateModel(
            training_configs=training_configs,
            training_y=training_y,
            kernel_fn=self.kernel,
            krr_alpha=1e-6,
        )

        best_y = np.min(training_y)
        n_to_acquire = n_to_sample - len(sampled_configs)

        candidate_configs = []
        candidate_scores = []
        candidate_tuples = []  # Keep full tuples for final preparation

        for c in range(self.acqu_sampling_density):
            # Sample a random config from the space
            print(f"Sampling candidate {c+1:<{len(str(self.acqu_sampling_density))}}/{self.acqu_sampling_density}. ", end="\r" if c < self.acqu_sampling_density - 1 else "", flush=True)
            candidate = resolve(
                pipeline=self._pipeline,
                domain_sampler=self._random_sampler,
                environment_values={},
            )

            # Extract config dict from resolution context
            config_dict = candidate[1].samplings_made

            # Predict objective and uncertainty at this config
            pred_mean, pred_var = surrogate.predict(config_dict)

            candidate_configs.append(config_dict)
            candidate_tuples.append(candidate)
            candidate_scores.append((pred_mean, pred_var))


        # Select top-k candidates with highest acquisition scores
        if self.use_batch_acquisition:
            # Compute batch-aware acquisition (greedy, accounts for candidate correlation)
            pred_means = np.array([score[0] for score in candidate_scores])
            pred_vars = np.array([score[1] for score in candidate_scores])
            acqu_scores = _greedy_batch_acquisition(
                pred_means, pred_vars, best_y,
                kernel_fn=self.kernel,
                candidate_configs=candidate_configs,
                acqu_func=self.acqu_function,  # Pass the configured acquisition function
            )
        else:
            # Standard acquisition: evaluate each candidate independently
            acqu_scores = np.array([
                self.acqu_function(score[0], score[1], best_y)
                for score in candidate_scores
            ])
        top_k_indices = np.argsort(acqu_scores)[-n_to_acquire:]

        # Return full tuples for final preparation (not just config dicts)
        selected_tuples = [candidate_tuples[i] for i in top_k_indices]

        return _prepare_sampled_configs(selected_tuples, max_trial_id, n_to_sample==1)

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        max_trial_id = _get_max_trial_id(trials)
        return [
            ImportedConfig(
                id=str(i),
                config=copy.deepcopy(config),
                result=copy.deepcopy(result),
            )
            for i, (config, result) in enumerate(
                external_evaluations, start=max_trial_id + 1
            )
        ]
