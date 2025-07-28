"""This module implements a simple random search optimizer for a NePS pipeline.
It samples configurations randomly from the pipeline's domain and environment values.
"""

from __future__ import annotations

import heapq
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from neps.space.neps_spaces.neps_space import _prepare_sampled_configs, resolve
from neps.space.neps_spaces.parameters import Float, Integer
from neps.space.neps_spaces.sampling import (
    CrossoverByMixingSampler,
    CrossoverNotPossibleError,
    MutatateUsingCentersSampler,
    MutateByForgettingSampler,
    PriorOrFallbackSampler,
    RandomSampler,
)

if TYPE_CHECKING:
    import neps.state.optimizer as optimizer_state
    import neps.state.trial as trial_state
    from neps.optimizers import optimizer
    from neps.space.neps_spaces.parameters import PipelineSpace
    from neps.state.trial import Trial


@dataclass
class NePSRandomSearch:
    """A simple random search optimizer for a NePS pipeline.
    It samples configurations randomly from the pipeline's domain and environment values.

    Args:
        pipeline: The pipeline to optimize, which should be a Pipeline object.

    Raises:
        ValueError: If the pipeline is not a Pipeline object.
    """

    def __init__(
        self,
        pipeline: PipelineSpace,
        use_priors: bool = False,  # noqa: FBT001, FBT002
        ignore_fidelity: bool | Literal["highest fidelity"] = False,  # noqa: FBT002
    ):
        """Initialize the RandomSearch optimizer with a pipeline.

        Args:
            pipeline: The pipeline to optimize, which should be a Pipeline object.

        Raises:
            ValueError: If the pipeline is not a Pipeline object.
        """
        self._pipeline = pipeline

        self._environment_values = {}
        fidelity_attrs = self._pipeline.fidelity_attrs
        for fidelity_name, fidelity_obj in fidelity_attrs.items():
            if ignore_fidelity == "highest fidelity":
                self._environment_values[fidelity_name] = fidelity_obj.max_value
            elif not ignore_fidelity:
                raise ValueError(
                    "RandomSearch does not support fidelities by default. Consider using"
                    " a different optimizer or setting `ignore_fidelity=True` or `highest"
                    " fidelity`."
                )
            # Sample randomly from the fidelity bounds.
            elif isinstance(fidelity_obj._domain, Integer):
                assert isinstance(fidelity_obj.min_value, int)
                assert isinstance(fidelity_obj.max_value, int)
                self._environment_values[fidelity_name] = random.randint(
                    fidelity_obj.min_value, fidelity_obj.max_value
                )
            elif isinstance(fidelity_obj._domain, Float):
                self._environment_values[fidelity_name] = random.uniform(
                    fidelity_obj.min_value, fidelity_obj.max_value
                )

        self._random_sampler = RandomSampler(predefined_samplings={})
        self.use_prior = use_priors
        self._prior_sampler = PriorOrFallbackSampler(
            fallback_sampler=self._random_sampler
        )

    def __call__(
        self,
        trials: Mapping[str, trial_state.Trial],
        budget_info: optimizer_state.BudgetInfo | None,
        n: int | None = None,
    ) -> optimizer.SampledConfig | list[optimizer.SampledConfig]:
        """Sample configurations randomly from the pipeline's domain and environment
        values.

        Args:
            trials: A mapping of trial IDs to Trial objects, representing previous
                trials.
            budget_info: The budget information for the optimization process.
            n: The number of configurations to sample. If None, a single configuration
                will be sampled.

        Returns:
            A SampledConfig object or a list of SampledConfig objects, depending
                on the value of n.

        Raises:
            ValueError: If the pipeline is not a Pipeline object or if the trials are
                not a valid mapping of trial IDs to Trial objects.
        """
        n_prev_trials = len(trials)
        n_requested = 1 if n is None else n
        return_single = n is None

        if self.use_prior:
            chosen_pipelines = [
                resolve(
                    pipeline=self._pipeline,
                    domain_sampler=self._prior_sampler,
                    environment_values=self._environment_values,
                )
                for _ in range(n_requested)
            ]
        else:
            chosen_pipelines = [
                resolve(
                    pipeline=self._pipeline,
                    domain_sampler=self._random_sampler,
                    environment_values=self._environment_values,
                )
                for _ in range(n_requested)
            ]

        return _prepare_sampled_configs(chosen_pipelines, n_prev_trials, return_single)


@dataclass
class NePSComplexRandomSearch:
    """A complex random search optimizer for a NePS pipeline.
    It samples configurations randomly from the pipeline's domain and environment values,
    and also performs mutations and crossovers based on previous successful trials.

    Args:
        pipeline: The pipeline to optimize, which should be a Pipeline object.

    Raises:
        ValueError: If the pipeline is not a Pipeline object.
    """

    def __init__(
        self,
        pipeline: PipelineSpace,
        ignore_fidelity: bool | Literal["highest fidelity"] = False,  # noqa: FBT002
    ):
        """Initialize the ComplexRandomSearch optimizer with a pipeline.

        Args:
            pipeline: The pipeline to optimize, which should be a Pipeline object.

        Raises:
            ValueError: If the pipeline is not a Pipeline object.
        """
        self._pipeline = pipeline

        self._environment_values = {}
        fidelity_attrs = self._pipeline.fidelity_attrs
        for fidelity_name, fidelity_obj in fidelity_attrs.items():
            if ignore_fidelity == "highest fidelity":
                self._environment_values[fidelity_name] = fidelity_obj.max_value
            elif not ignore_fidelity:
                raise ValueError(
                    "ComplexRandomSearch does not support fidelities by default. Consider"
                    " using a different optimizer or setting `ignore_fidelity=True` or"
                    " `highest fidelity`."
                )
            # Sample randomly from the fidelity bounds.
            elif isinstance(fidelity_obj._domain, Integer):
                assert isinstance(fidelity_obj.min_value, int)
                assert isinstance(fidelity_obj.max_value, int)
                self._environment_values[fidelity_name] = random.randint(
                    fidelity_obj.min_value, fidelity_obj.max_value
                )
            elif isinstance(fidelity_obj._domain, Float):
                self._environment_values[fidelity_name] = random.uniform(
                    fidelity_obj.min_value, fidelity_obj.max_value
                )

        self._random_sampler = RandomSampler(
            predefined_samplings={},
        )
        self._try_always_priors_sampler = PriorOrFallbackSampler(
            fallback_sampler=self._random_sampler,
            always_use_prior=True,
        )
        self._sometimes_priors_sampler = PriorOrFallbackSampler(
            fallback_sampler=self._random_sampler
        )
        self._n_top_trials = 5

    def __call__(
        self,
        trials: Mapping[str, trial_state.Trial],
        budget_info: optimizer_state.BudgetInfo | None,
        n: int | None = None,
    ) -> optimizer.SampledConfig | list[optimizer.SampledConfig]:
        """Sample configurations randomly from the pipeline's domain and environment
        values, and also perform mutations and crossovers based on previous successful
        trials.

        Args:
            trials: A mapping of trial IDs to Trial objects, representing previous
                trials.
            budget_info: The budget information for the optimization process.
            n: The number of configurations to sample. If None, a single configuration
                will be sampled.

        Returns:
            A SampledConfig object or a list of SampledConfig objects, depending
                on the value of n.

        Raises:
            ValueError: If the pipeline is not a Pipeline object or if the trials are
                not a valid mapping of trial IDs to Trial objects.
        """
        n_prev_trials = len(trials)
        n_requested = 1 if n is None else n
        return_single = n is None

        random_pipelines = [
            resolve(
                pipeline=self._pipeline,
                domain_sampler=self._random_sampler,
                environment_values=self._environment_values,
            )
            for _ in range(n_requested * 5)
        ]
        sometimes_priors_pipelines = [
            resolve(
                pipeline=self._pipeline,
                domain_sampler=self._sometimes_priors_sampler,
                environment_values=self._environment_values,
            )
            for _ in range(n_requested * 5)
        ]

        mutated_incumbents = []
        crossed_over_incumbents = []

        successful_trials: list[Trial] = list(
            filter(
                lambda trial: (
                    trial.report.reported_as == trial.State.SUCCESS
                    if trial.report is not None
                    else False
                ),
                trials.values(),
            )
        )
        if len(successful_trials) > 0:
            self._n_top_trials = 5
            top_trials = heapq.nsmallest(
                self._n_top_trials,
                successful_trials,
                key=lambda trial: (
                    float(trial.report.objective_to_minimize)
                    if trial.report
                    and isinstance(trial.report.objective_to_minimize, float)
                    else float("inf")
                ),
            )  # Will have up to `self._n_top_trials` items.

            # Do some mutations.
            for top_trial in top_trials:
                top_trial_config = top_trial.config

                # Mutate by resampling around some values of the original config.
                mutated_incumbents += [
                    resolve(
                        pipeline=self._pipeline,
                        domain_sampler=MutatateUsingCentersSampler(
                            predefined_samplings=top_trial_config,
                            n_mutations=1,
                        ),
                        environment_values=self._environment_values,
                    )
                    for _ in range(n_requested * 5)
                ]
                mutated_incumbents += [
                    resolve(
                        pipeline=self._pipeline,
                        domain_sampler=MutatateUsingCentersSampler(
                            predefined_samplings=top_trial_config,
                            n_mutations=max(
                                1, random.randint(1, int(len(top_trial_config) / 2))
                            ),
                        ),
                        environment_values=self._environment_values,
                    )
                    for _ in range(n_requested * 5)
                ]

                # Mutate by completely forgetting some values of the original config.
                mutated_incumbents += [
                    resolve(
                        pipeline=self._pipeline,
                        domain_sampler=MutateByForgettingSampler(
                            predefined_samplings=top_trial_config,
                            n_forgets=1,
                        ),
                        environment_values=self._environment_values,
                    )
                    for _ in range(n_requested * 5)
                ]
                mutated_incumbents += [
                    resolve(
                        pipeline=self._pipeline,
                        domain_sampler=MutateByForgettingSampler(
                            predefined_samplings=top_trial_config,
                            n_forgets=max(
                                1, random.randint(1, int(len(top_trial_config) / 2))
                            ),
                        ),
                        environment_values=self._environment_values,
                    )
                    for _ in range(n_requested * 5)
                ]

            # Do some crossovers.
            if len(top_trials) > 1:
                for _ in range(n_requested * 3):
                    trial_1, trial_2 = random.sample(top_trials, k=2)

                    try:
                        crossover_sampler = CrossoverByMixingSampler(
                            predefined_samplings_1=trial_1.config,
                            predefined_samplings_2=trial_2.config,
                            prefer_first_probability=0.5,
                        )
                    except CrossoverNotPossibleError:
                        # A crossover was not possible for them. Do nothing.
                        pass
                    else:
                        crossed_over_incumbents.append(
                            resolve(
                                pipeline=self._pipeline,
                                domain_sampler=crossover_sampler,
                                environment_values=self._environment_values,
                            ),
                        )

                    try:
                        crossover_sampler = CrossoverByMixingSampler(
                            predefined_samplings_1=trial_2.config,
                            predefined_samplings_2=trial_1.config,
                            prefer_first_probability=0.5,
                        )
                    except CrossoverNotPossibleError:
                        # A crossover was not possible for them. Do nothing.
                        pass
                    else:
                        crossed_over_incumbents.append(
                            resolve(
                                pipeline=self._pipeline,
                                domain_sampler=crossover_sampler,
                                environment_values=self._environment_values,
                            ),
                        )

        all_sampled_pipelines = [
            *random_pipelines,
            *sometimes_priors_pipelines,
            *mutated_incumbents,
            *crossed_over_incumbents,
        ]

        # Here we can have a model which picks from all the sampled pipelines.
        # Currently, we just pick randomly from them.
        chosen_pipelines = random.sample(all_sampled_pipelines, k=n_requested)

        if n_prev_trials == 0:
            # In this case, always include the prior pipeline.
            prior_pipeline = resolve(
                pipeline=self._pipeline,
                domain_sampler=self._try_always_priors_sampler,
                environment_values=self._environment_values,
            )
            chosen_pipelines[0] = prior_pipeline

        return _prepare_sampled_configs(chosen_pipelines, n_prev_trials, return_single)
