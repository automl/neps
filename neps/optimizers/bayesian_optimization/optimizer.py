from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import torch
from botorch.acquisition import LinearMCObjective
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.bayesian_optimization.models.gp import (
    encode_trials_for_gp,
    fit_and_acquire_from_gp,
    make_default_single_obj_gp,
)
from neps.optimizers.initial_design import make_initial_design
from neps.sampling import Prior
from neps.search_spaces.encoding import ConfigEncoder

if TYPE_CHECKING:
    from neps.search_spaces import SearchSpace
    from neps.state import BudgetInfo, Trial


def _pibo_exp_term(
    n_sampled_already: int,
    ndims: int,
    initial_design_size: int,
) -> float:
    # pibo paper
    # https://arxiv.org/pdf/2204.11051
    #
    # they use some constant determined from max problem budget. seems impractical,
    # given we might not know the final budget (i.e. imagine you iteratively increase
    # the budget as you go along).
    #
    # instead, we base it on the fact that in lower dimensions, we don't to rely
    # on the prior for too long as the amount of space you need to cover around the
    # prior is fairly low. effectively, since the gp needs little samples to
    # model pretty effectively in low dimension, we can derive the utility from
    # the prior pretty quickly.
    #
    # however, for high dimensional settings, we want to rely longer on the prior
    # for longer as the number of samples needed to model the area around the prior
    # is much larger, and deriving the utility will take longer.
    #
    # in the end, we would like some curve going from 1->0 as n->inf, where `n` is
    # the number of samples we have done so far.
    # the easiest function that does this is `exp(-n)`, with some discounting of `n`
    # dependant on the number of dimensions.
    n_bo_samples = n_sampled_already - initial_design_size
    return math.exp(-n_bo_samples / ndims)


class BayesianOptimization(BaseOptimizer):
    """Implements the basic BO loop."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        *,
        initial_design_size: int | None = None,
        use_priors: bool = False,
        use_cost: bool = False,
        cost_on_log_scale: bool = True,
        sample_prior_first: bool = False,
        device: torch.device | None = None,
        encoder: ConfigEncoder | None = None,
        seed: int | None = None,
        max_cost_total: Any | None = None,  # TODO: remove
        surrogate_model: Any | None = None,  # TODO: remove
        objective_to_minimize_value_on_error: Any | None = None,  # TODO: remove
        cost_value_on_error: Any | None = None,  # TODO: remove
        ignore_errors: Any | None = None,  # TODO: remove
    ):
        """Initialise the BO loop.

        Args:
            pipeline_space: Space in which to search
            initial_design_size: Number of samples used before using the surrogate model.
                If None, it will use the number of parameters in the search space.
            use_priors: Whether to use priors set on the hyperparameters during search.
            use_cost: Whether to consider reported "cost" from configurations in decision
                making. If True, the optimizer will weigh potential candidates by how much
                they cost, incentivising the optimizer to explore cheap, good performing
                configurations. This amount is modified over time

                !!! warning

                    If using `cost`, cost must be provided in the reports of the trials.

            cost_on_log_scale: Whether to use the log of the cost when using cost.
            sample_prior_first: Whether to sample the default configuration first.
            seed: Seed to use for the random number generator of samplers.
            device: Device to use for the optimization.
            encoder: Encoder to use for encoding the configurations. If None, it will
                will use the default encoder.

        Raises:
            ValueError: if initial_design_size < 1
            ValueError: if no kernel is provided
        """
        if seed is not None:
            raise NotImplementedError(
                "Seed is not implemented yet for BayesianOptimization"
            )
        if any(pipeline_space.graphs):
            raise NotImplementedError("Only supports flat search spaces for now!")
        if any(pipeline_space.fidelities):
            raise ValueError(
                "Fidelities are not supported for BayesianOptimization."
                " Please consider setting the fidelity to a constant value."
                f" Got: {pipeline_space.fidelities}"
            )

        super().__init__(pipeline_space=pipeline_space)

        self.encoder = encoder or ConfigEncoder.from_space(
            space=pipeline_space,
            include_constants_when_decoding=True,
        )
        self.prior = Prior.from_space(pipeline_space) if use_priors is True else None
        self.use_cost = use_cost
        self.use_priors = use_priors
        self.cost_on_log_scale = cost_on_log_scale
        self.device = device
        self.sample_prior_first = sample_prior_first

        if initial_design_size is not None:
            self.n_initial_design = initial_design_size
        else:
            self.n_initial_design = len(pipeline_space.numerical) + len(
                pipeline_space.categoricals
            )

    @override
    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        _n = 1 if n is None else n
        n_sampled = len(trials)
        config_ids = iter(str(i + 1) for i in range(n_sampled, n_sampled + _n))
        space = self.pipeline_space

        sampled_configs: list[SampledConfig] = []

        # If the amount of configs evaluated is less than the initial design
        # requirement, keep drawing from initial design
        n_evaluated = sum(
            1
            for trial in trials.values()
            if trial.report is not None and trial.report.loss is not None
        )
        if n_evaluated < self.n_initial_design:
            design_samples = make_initial_design(
                space=space,
                encoder=self.encoder,
                sample_prior_first=self.sample_prior_first if n_sampled == 0 else False,
                sampler=self.prior if self.prior is not None else "uniform",
                seed=None,  # TODO: Seeding
                sample_size=_n,
                sample_fidelity="max",
            )

            sampled_configs.extend(
                [
                    SampledConfig(id=config_id, config=config)
                    for config_id, config in zip(
                        config_ids,
                        design_samples,
                        strict=False,
                    )
                ]
            )
            if len(sampled_configs) == _n:
                if n is None:
                    return sampled_configs[0]

                return sampled_configs

        # Otherwise, we encode trials and setup to fit and acquire from a GP
        data, encoder = encode_trials_for_gp(
            trials, space, device=self.device, encoder=self.encoder
        )

        cost_percent = None
        if self.use_cost:
            if max_cost_total_info is None:
                raise ValueError(
                    "Must provide a 'cost' to configurations if using cost"
                    " with BayesianOptimization."
                )
            if max_cost_total_info.max_cost_total is None:
                raise ValueError("Cost budget must be set if using cost")
            cost_percent = (
                max_cost_total_info.used_cost_budget / max_cost_total_info.max_cost_total
            )

        # If we should use the prior, weight the acquisition function by
        # the probability of it being sampled from the prior.
        pibo_exp_term = None
        prior = None
        if self.prior:
            pibo_exp_term = _pibo_exp_term(
                n_sampled, encoder.ncols, self.n_initial_design
            )
            # If the exp term is insignificant, skip prior acq. weighting
            prior = None if pibo_exp_term < 1e-4 else self.prior

        gp = make_default_single_obj_gp(x=data.x, y=data.y, encoder=encoder)
        candidates = fit_and_acquire_from_gp(
            gp=gp,
            x_train=data.x,
            encoder=encoder,
            acquisition=qLogNoisyExpectedImprovement(
                model=gp,
                X_baseline=data.x,
                # Unfortunatly, there's no option to indicate that we minimize
                # the AcqFunction so we need to do some kind of transformation.
                # https://github.com/pytorch/botorch/issues/2316#issuecomment-2085964607
                objective=LinearMCObjective(weights=torch.tensor([-1.0])),
                X_pending=data.x_pending,
                prune_baseline=True,
            ),
            prior=prior,
            n_candidates_required=_n,
            pibo_exp_term=pibo_exp_term,
            costs=data.cost if self.use_cost else None,
            cost_percentage_used=cost_percent,
            costs_on_log_scale=self.cost_on_log_scale,
        )

        configs = encoder.decode(candidates)
        sampled_configs.extend(
            [
                SampledConfig(id=config_id, config=config)
                for config_id, config in zip(config_ids, configs, strict=True)
            ]
        )

        if n is None:
            return sampled_configs[0]

        return sampled_configs
