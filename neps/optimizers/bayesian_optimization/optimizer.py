from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import torch
from botorch.acquisition import LinearMCObjective
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Log, Standardize
from gpytorch import ExactMarginalLogLikelihood

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.bayesian_optimization.acquisition_functions.cost_cooling import (
    cost_cooled_acq,
)
from neps.optimizers.bayesian_optimization.acquisition_functions.pibo import (
    pibo_acquisition,
)
from neps.optimizers.bayesian_optimization.models.gp import (
    make_default_single_obj_gp,
    optimize_acq,
)
from neps.optimizers.intial_design import make_initial_design
from neps.sampling import Prior
from neps.search_spaces.encoding import TensorEncoder
from neps.search_spaces.hyperparameters.categorical import CategoricalParameter

if TYPE_CHECKING:
    from neps.search_spaces import SearchSpace
    from neps.search_spaces.hyperparameters.float import FloatParameter
    from neps.search_spaces.hyperparameters.integer import IntegerParameter
    from neps.state import BudgetInfo, Trial


def _missing_fill_strategy(
    y: torch.Tensor,
    strategy: Literal["mean", "worst", "3std", "nan"],
    *,
    lower_is_better: bool,
) -> torch.Tensor:
    # Assumes minimization
    if y.ndim != 1:
        raise ValueError("Only supports single objective optimization for now!")

    match strategy:
        case "nan":
            return y
        case "mean":
            return torch.nan_to_num(y, nan=y.mean().item())
        case "worst":
            worst = y.min() if lower_is_better else y.max()
            return torch.nan_to_num(y, nan=worst.item())
        case "3std":
            sign = 1 if lower_is_better else -1
            std = y.std()
            return torch.nan_to_num(y, nan=y.mean().item() + sign * 3 * std.item())
        case _:
            raise ValueError(f"Unknown strategy: {strategy}")


def _missing_y_strategy(y: torch.Tensor) -> torch.Tensor:
    # TODO: Figure out what to do if there's no reported loss value.
    # Some strategies:
    # 1. Replace with NaN, in which case GPYtorch ignores it
    #   * Good if crash is random crash, in which case we do not wish to model
    #   a performance because of it.
    # 2. Replace with worst value seen so far
    #   * Good if crash is systematic, in which case we wish to model it as
    #   basically, "don't go here" while remaining in the range of possible
    #   values for the GP.
    # 3. Replace with mean
    #   * Same as above but keeps the optimization of the GP landscape
    #   smoother. Good if we have a mix of non-systematic and systematic
    #   crashed. Likely the safest option as GP will likely be unconfident in
    #   unsystematic crash cases, especially if it seems like a rare-event.
    #   Will also unlikely be a candidate region if systematic and we observe
    #   a few crashes there. However would take longer to learn of systematic
    #   crash regions.
    return _missing_fill_strategy(y, strategy="mean", lower_is_better=True)


def _missing_cost_strategy(cost: torch.Tensor) -> torch.Tensor:
    # TODO: Figure out what to do if there's no reported cost value
    # Likely best to just fill in worst cost seen so far as this crash
    # cost us a lot of time and we do not want to waste time on this
    # region again. However if the crash was random, we might enter some
    # issues.
    return _missing_fill_strategy(cost, strategy="3std", lower_is_better=True)


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


def _cost_used_budget_percentage(budget_info: BudgetInfo) -> float:
    if budget_info.max_cost_budget is not None:
        return budget_info.used_cost_budget / budget_info.max_cost_budget

    raise ValueError("No cost budget provided!")


class BayesianOptimization(BaseOptimizer):
    """Implements the basic BO loop."""

    def __init__(  # noqa: D417
        self,
        pipeline_space: SearchSpace,
        *,
        initial_design_size: int | None = None,
        use_priors: bool = False,
        use_cost: bool = False,
        sample_default_first: bool = False,
        device: torch.device | None = None,
        encoder: TensorEncoder | None = None,
        seed: int | None = None,
        budget: Any | None = None,  # TODO: remove
        surrogate_model: Any | None = None,  # TODO: remove
        loss_value_on_error: Any | None = None,  # TODO: remove
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

            sample_default_first: Whether to sample the default configuration first.
            seed: Seed to use for the random number generator of samplers.
            device: Device to use for the optimization.
            encoder: Encoder to use for encoding the configurations. If None, it will
                will use the default encoder.

        Raises:
            ValueError: if initial_design_size < 1
            ValueError: if no kernel is provided
        """
        if any(pipeline_space.graphs):
            raise NotImplementedError("Only supports flat search spaces for now!")
        super().__init__(pipeline_space=pipeline_space)

        params: dict[str, CategoricalParameter | FloatParameter | IntegerParameter] = {
            **pipeline_space.numerical,
            **pipeline_space.categoricals,
        }
        self.encoder = TensorEncoder.default(params) if encoder is None else encoder
        self.prior = Prior.from_parameters(params) if use_priors is True else None
        self.seed = seed
        self.use_cost = use_cost
        self.device = device
        self.sample_default_first = sample_default_first
        self.n_initial_design = initial_design_size
        self.initial_design_: list[dict[str, Any]] | None = None

    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo,
        optimizer_state: dict[str, Any],
        seed: int | None = None,
    ) -> SampledConfig:
        if seed is not None:
            raise NotImplementedError(
                "Seed is not yet implemented for BayesianOptimization"
            )

        n_trials_sampled = len(trials)
        config_id = str(n_trials_sampled + 1)

        # If we havn't passed the intial design phase
        if self.initial_design_ is None:
            self.initial_design_ = make_initial_design(
                space=self.pipeline_space,
                encoder=self.encoder,
                sample_default_first=self.sample_default_first,
                sampler=self.prior if self.prior is not None else "sobol",
                seed=seed,
                sample_size=(
                    "ndim" if self.n_initial_design is None else self.n_initial_design
                ),
                sample_fidelity="max",
            )

        if n_trials_sampled < len(self.initial_design_):
            config = self.initial_design_[n_trials_sampled]
            return SampledConfig(id=config_id, config=config)

        # Now we actually do the BO loop, start by encoding the data
        # TODO: Lift this into runtime, let the optimizer advertise the encoding wants...
        x_configs: list[Mapping[str, Any]] = []
        ys: list[float] = []
        costs: list[float] = []
        pending: list[Mapping[str, Any]] = []
        for trial in trials.values():
            if trial.state.pending():
                pending.append(trial.config)
            else:
                assert trial.report is not None
                x_configs.append(trial.config)
                ys.append(
                    trial.report.loss if trial.report.loss is not None else torch.nan
                )
                if self.use_cost:
                    cost_z_score = trial.report.cost
                    costs.append(cost_z_score if cost_z_score is not None else torch.nan)

        x = self.encoder.pack(x_configs, device=self.device)
        maybe_x_pending_tensor = None
        if len(pending) > 0:
            x_pending = self.encoder.pack(pending, device=self.device)
            maybe_x_pending_tensor = x_pending.tensor

        y = torch.tensor(ys, dtype=torch.float64, device=self.device)
        y = _missing_y_strategy(y)

        # Now fit our model
        y_model = make_default_single_obj_gp(
            x,
            y,
            # TODO: We should consider applying some heurisitc to see if this should
            # also include a log transform, similar as we do to cost if using `use_cost`.
            y_transform=Standardize(m=1),
        )
        y_likelihood = y_model.likelihood

        fit_gpytorch_mll(
            ExactMarginalLogLikelihood(likelihood=y_likelihood, model=y_model)
        )

        # NOTE: We use:
        # * q - allows accounting for pending points, normally used to get a batch
        #       of points.
        # * log - More numerically stable
        # * Noisy - In Deep-Learning, we shouldn't take f.min() incase it was a noise
        #           spike. This accounts for noise in objective.
        # * ExpectedImprovement - Cause ya know, the default.
        acq = qLogNoisyExpectedImprovement(
            y_model,
            X_baseline=x.tensor,
            X_pending=maybe_x_pending_tensor,
            # Unfortunatly, there's no option to indicate that we minimize
            # the AcqFunction so we need to do some kind of transformation.
            # https://github.com/pytorch/botorch/issues/2316#issuecomment-2085964607
            objective=LinearMCObjective(weights=torch.tensor([-1.0])),
        )

        # If we should use the prior, weight the acquisition function by
        # the probability of it being sampled from the prior.
        if self.prior:
            pibo_exp_term = _pibo_exp_term(
                n_trials_sampled,
                self.encoder.ncols,
                len(self.initial_design_),
            )

            # If the amount of weight derived from the pibo exponent becomes
            # insignificant, we don't use it as it as it adds extra computational
            # burden and introduces more chance of numerical instability.
            significant_lower_bound = 1e-4
            if pibo_exp_term > significant_lower_bound:
                acq = pibo_acquisition(
                    acq,
                    prior=self.prior,
                    prior_exponent=pibo_exp_term,
                    x_domain=self.encoder.domains,
                    X_pending=maybe_x_pending_tensor,
                )

        # If we should use cost, weight the acquisition function by the cost
        # of the configurations.
        if self.use_cost:
            cost = torch.tensor(costs, dtype=torch.float64, device=self.device)
            cost_z_score = _missing_cost_strategy(cost)

            cost_model = make_default_single_obj_gp(
                x,
                cost_z_score,
                y_transform=ChainedOutcomeTransform(
                    # TODO: Maybe some way for a user to specify their cost
                    # is on a log scale?
                    log=Log(),
                    standardize=Standardize(m=1),
                ),
            )
            cost_likelihood = cost_model.likelihood

            # Optimize the cost model
            fit_gpytorch_mll(
                ExactMarginalLogLikelihood(likelihood=cost_likelihood, model=cost_model)
            )
            acq = cost_cooled_acq(
                acq_fn=acq,
                model=cost_model,
                used_budget_percentage=_cost_used_budget_percentage(budget_info),
                X_pending=maybe_x_pending_tensor,
            )

        # Finally, optimize the acquisition function to get a configuration
        candidates, _eis = optimize_acq(acq_fn=acq, encoder=self.encoder, acq_options={})

        assert len(candidates) == 1, "Expected only one candidate!"
        config = self.encoder.unpack(candidates)[0]

        return SampledConfig(id=config_id, config=config)
