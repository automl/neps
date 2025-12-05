"""This module implements a Regularized Evolution optimizer for NEPS."""

from __future__ import annotations

import heapq
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from neps.optimizers.optimizer import ImportedConfig
from neps.space.neps_spaces.neps_space import (
    SamplingResolutionContext,
    _prepare_sampled_configs,
    resolve,
)
from neps.space.neps_spaces.parameters import Float, Integer, PipelineSpace
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
    from neps.state.pipeline_eval import UserResultDict
    from neps.state.trial import Trial


@dataclass
class NePSRegularizedEvolution:
    """A Regularized Evolution optimizer for a NePS pipeline.
    It samples configurations based on mutations and crossovers of previous successful
    trials, using a tournament selection mechanism.

    Args:
        pipeline: The pipeline to optimize.
        population_size: The size of the population for evolution.
        tournament_size: The size of the tournament for selecting parents.
        use_priors: Whether to use priors when sampling if available.
        mutation_type: The type of mutation to use (e.g., "mutate_best",
            "crossover_top_2"). If a float is provided, it is interpreted as the
            probability of choosing mutation, compared to the probability of crossover.
        n_mutations: The number of mutations to apply. A fixed integer, "random" for
            a random number between 1 and half the parameters, or "half" to mutate
            half the parameters.
        n_forgets: The number of parameters to forget. A fixed integer, "random" for
            a random number between 1 and half the parameters, or "half" to forget
            half the parameters.
        ignore_fidelity: Whether to ignore fidelity when sampling. If set to "highest
            fidelity", the highest fidelity values will be used. If True, fidelity
            values will be sampled randomly.
    """

    def __init__(
        self,
        pipeline: PipelineSpace,
        population_size: int = 20,
        tournament_size: int = 5,
        use_priors: bool = True,  # noqa: FBT001, FBT002
        mutation_type: float | Literal["mutate_best", "crossover_top_2"] = 0.5,
        n_mutations: int | Literal["random", "half"] | None = "random",
        n_forgets: int | Literal["random", "half"] | None = None,
        ignore_fidelity: bool | Literal["highest fidelity"] = False,  # noqa: FBT002
    ):
        """Initialize the RegularizedEvolution optimizer with a pipeline.

        Args:
            pipeline: The pipeline to optimize, which should be a Pipeline object.
            population_size: The size of the population for evolution.
            tournament_size: The size of the tournament for selecting parents.
            use_priors: Whether to use priors when sampling if available.
            mutation_type: The type of mutation to use (e.g., "mutate_best",
                "crossover_top_2"). If a float is provided, it is interpreted as the
                probability of choosing mutation, compared to the probability of
                crossover.
            n_mutations: The number of mutations to apply. A fixed integer, "random" for
                a random number between 1 and half the parameters, or "half" to mutate
                half the parameters.
            n_forgets: The number of parameters to forget. A fixed integer, "random" for
                a random number between 1 and half the parameters, or "half" to forget
                half the parameters.
            ignore_fidelity: Whether to ignore fidelity when sampling. If set to "highest
                fidelity", the highest fidelity values will be used. If True, fidelity
                values will be sampled randomly.

        Raises:
            ValueError: If the pipeline is not a Pipeline object.
        """
        self._pipeline = pipeline

        self._random_or_prior_sampler: RandomSampler | PriorOrFallbackSampler = (
            RandomSampler(
                predefined_samplings={},
            )
        )
        if use_priors:
            self._random_or_prior_sampler = PriorOrFallbackSampler(
                fallback_sampler=self._random_or_prior_sampler
            )
        assert population_size >= tournament_size, (
            "Population size must be greater than or equal to tournament size."
        )
        self._tournament_size = tournament_size
        self._population_size = population_size
        self._mutation_type = mutation_type
        self._n_mutations = n_mutations
        self._n_forgets = n_forgets
        self._ignore_fidelity = ignore_fidelity

    def _mutate_best(
        self, top_trial_config: Mapping[str, Any]
    ) -> tuple[PipelineSpace, SamplingResolutionContext]:
        """Mutate the best trial's config by resampling or forgetting parameters.

        Args:
            top_trial_config: The configuration of the best trial to mutate.

        Returns:
            A mutated configuration (PipelineSpace and context tuple).

        Raises:
            ValueError: If both n_mutations and n_forgets are None.
        """
        if self._n_mutations:
            n_mut = (
                self._n_mutations
                if isinstance(self._n_mutations, int)
                else (
                    random.randint(1, len(top_trial_config) // 2)
                    if self._n_mutations == "random"
                    else len(top_trial_config) // 2
                )
            )
            return resolve(
                pipeline=self._pipeline,
                domain_sampler=MutatateUsingCentersSampler(
                    predefined_samplings=top_trial_config,
                    n_mutations=n_mut,
                ),
                environment_values=self.sampled_fidelity_values(),
            )
        if self._n_forgets:
            n_forg = (
                self._n_forgets
                if isinstance(self._n_forgets, int)
                else (
                    random.randint(1, len(top_trial_config) // 2)
                    if self._n_forgets == "random"
                    else max(1, len(top_trial_config) // 2)
                )
            )
            return resolve(
                pipeline=self._pipeline,
                domain_sampler=MutateByForgettingSampler(
                    predefined_samplings=top_trial_config,
                    n_forgets=n_forg,
                ),
                environment_values=self.sampled_fidelity_values(),
            )
        raise ValueError("At least one of n_mutations or n_forgets must not be None.")

    def _crossover_top_2(
        self, sorted_trials: list[Trial]
    ) -> tuple[PipelineSpace, SamplingResolutionContext]:
        """Perform crossover between top trials from the tournament.

        Args:
            sorted_trials: List of configurations sorted by objective (best first).

        Returns:
            A configuration created by crossover (PipelineSpace and context tuple),
            or a mutated config if crossover fails.
        """
        # Create all possible crossovers between the top trials, sorted by smallest
        # combined index.
        all_crossovers = [
            (x, y)
            for x in range(len(sorted_trials))
            for y in range(len(sorted_trials))
            if x < y
        ]
        all_crossovers.sort(key=lambda pair: pair[0] + pair[1])

        for n, (config_1, config_2) in enumerate(all_crossovers):
            top_trial_config = sorted_trials[config_1].config
            second_best_trial_config = sorted_trials[config_2].config

            # Crossover between the best two trials' configs to create a new config.
            try:
                crossover_sampler = CrossoverByMixingSampler(
                    predefined_samplings_1=top_trial_config,
                    predefined_samplings_2=second_best_trial_config,
                    prefer_first_probability=0.5,
                )
            except CrossoverNotPossibleError:
                # A crossover was not possible for them. Increase configs and try again.
                # If we have tried all crossovers, mutate the best instead.
                if n == len(all_crossovers) - 1:
                    # Mutate 50% of the top trial's config.
                    return resolve(
                        pipeline=self._pipeline,
                        domain_sampler=MutatateUsingCentersSampler(
                            predefined_samplings=top_trial_config,
                            n_mutations=max(1, int(len(top_trial_config) / 2)),
                        ),
                        environment_values=self.sampled_fidelity_values(),
                    )
                continue
            else:
                return resolve(
                    pipeline=self._pipeline,
                    domain_sampler=crossover_sampler,
                    environment_values=self.sampled_fidelity_values(),
                )

        # Fallback in case all crossovers fail (shouldn't happen, but be safe)
        return self._mutate_best(sorted_trials[0].config)

    def sampled_fidelity_values(
        self,
    ) -> Mapping[str, float | int]:
        """Get the sampled fidelity values used in the optimizer.

        Returns:
            A mapping of fidelity names to their sampled values.
        """

        environment_values = {}
        fidelity_attrs = self._pipeline.fidelity_attrs
        for fidelity_name, fidelity_obj in fidelity_attrs.items():
            # If the user specifically asked for the highest fidelity, use that.
            if self._ignore_fidelity == "highest fidelity":
                environment_values[fidelity_name] = fidelity_obj.upper
            # If the user asked to ignore fidelities, sample a value randomly from the
            # domain.
            elif self._ignore_fidelity is True:
                # Sample randomly from the fidelity bounds.
                if isinstance(fidelity_obj.domain, Integer):
                    assert isinstance(fidelity_obj.lower, int)
                    assert isinstance(fidelity_obj.upper, int)
                    environment_values[fidelity_name] = random.randint(
                        fidelity_obj.lower, fidelity_obj.upper
                    )
                elif isinstance(fidelity_obj.domain, Float):
                    environment_values[fidelity_name] = random.uniform(
                        fidelity_obj.lower, fidelity_obj.upper
                    )
            # By default we don't support fidelities unless explicitly requested.
            else:
                raise ValueError(
                    "RegularizedEvolution does not support fidelities by default. "
                    "Consider using a different optimizer or setting "
                    "`ignore_fidelity=True` or `highest fidelity`."
                )
        return environment_values

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

        if n_prev_trials < self._population_size:
            # Just do random sampling until we have enough trials.
            random_pipelines = [
                resolve(
                    pipeline=self._pipeline,
                    domain_sampler=self._random_or_prior_sampler,
                    environment_values=self.sampled_fidelity_values(),
                )
                for _ in range(n_requested)
            ]

            return _prepare_sampled_configs(
                random_pipelines, n_prev_trials, n_requested == 1
            )

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

        # If we have no successful trials yet, fall back to random sampling.
        if len(successful_trials) == 0:
            random_pipelines = [
                resolve(
                    pipeline=self._pipeline,
                    domain_sampler=self._random_or_prior_sampler,
                    environment_values=self.sampled_fidelity_values(),
                )
                for _ in range(n_requested)
            ]

            return _prepare_sampled_configs(
                random_pipelines, n_prev_trials, n_requested == 1
            )

        return_pipelines = []

        for _ in range(n_requested):
            # Select the most recent trials to form the tournament.
            # We want the last (most recent) self._population_size successful trials.
            latest_trials = heapq.nlargest(
                self._population_size,
                successful_trials,
                key=lambda trial: (
                    trial.metadata.time_end
                    if trial.metadata and isinstance(trial.metadata.time_end, float)
                    else 0.0
                ),
            )

            tournament_trials = [
                random.sample((latest_trials), k=1)[0]
                for _ in range(min(self._tournament_size, len(latest_trials)))
            ]

            # Sort the tournament by objective and pick the best as the parent.
            def _obj_key(trial: Trial) -> float:
                return (
                    float(trial.report.objective_to_minimize)
                    if trial.report
                    and isinstance(trial.report.objective_to_minimize, float)
                    else float("inf")
                )

            sorted_trials = sorted(tournament_trials, key=_obj_key)

            top_trial_config = sorted_trials[0].config

            # Mutate or crossover the best trial's config to create a new config.
            if self._mutation_type == "mutate_best":
                mutated_incumbent = self._mutate_best(top_trial_config)
                return_pipelines.append(mutated_incumbent)
            elif self._mutation_type == "crossover_top_2":
                crossed_over_incumbent = self._crossover_top_2(sorted_trials)
                return_pipelines.append(crossed_over_incumbent)
            elif isinstance(self._mutation_type, float):
                if self._mutation_type < 0.0 or self._mutation_type > 1.0:
                    raise ValueError(
                        f"Invalid mutation probability: {self._mutation_type}. "
                        "It must be between 0.0 and 1.0."
                    )
                rand_val = random.random()

                if rand_val < self._mutation_type:
                    return_pipelines.append(self._mutate_best(top_trial_config))
                else:
                    return_pipelines.append(self._crossover_top_2(sorted_trials))
            else:
                raise ValueError(f"Invalid mutation type: {self._mutation_type}")

        return _prepare_sampled_configs(return_pipelines, n_prev_trials, n_requested == 1)

    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        """Import external evaluations as trials.

        Args:
            external_evaluations: A sequence of tuples containing configurations and
                their evaluation results.
            trials: A mapping of trial IDs to Trial objects, representing existing
                trials.

        Returns:
            A list of ImportedConfig objects representing the imported trials.
        """
        n_trials = len(trials)
        imported_configs = []
        for i, (config, result) in enumerate(external_evaluations):
            config_id = str(n_trials + i + 1)
            imported_configs.append(
                ImportedConfig(
                    config=config,
                    id=config_id,
                    result=result,
                )
            )
        return imported_configs
