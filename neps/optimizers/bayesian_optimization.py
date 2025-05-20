from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from botorch.acquisition import LinearMCObjective
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.transforms.outcome import Standardize

from neps.optimizers.models.gp import (
    encode_trials_for_gp,
    fit_and_acquire_from_gp,
    make_default_single_obj_gp,
)
from neps.optimizers.optimizer import SampledConfig
from neps.optimizers.utils.initial_design import make_initial_design

if TYPE_CHECKING:
    from neps.sampling import Prior
    from neps.space import ConfigEncoder, SearchSpace
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


@dataclass
class BayesianOptimization:
    """Uses `botorch` as an engine for doing bayesian optimiziation."""

    space: SearchSpace
    """The search space to use."""

    encoder: ConfigEncoder
    """The encoder to use for encoding and decoding configurations."""

    prior: Prior | None
    """The prior to use for sampling configurations and inferring their likelihood."""

    sample_prior_first: bool
    """Whether to sample the prior configuration first."""

    cost_aware: bool | Literal["log"]
    """Whether to consider the cost of configurations in decision making."""

    n_initial_design: int
    """The number of initial design samples to use before fitting the GP."""

    device: torch.device | None
    """The device to use for the optimization."""

    reference_point: tuple[float, ...] | None = None
    """The reference point to use for the multi-objective optimization."""

    def __call__(  # noqa: C901, PLR0912, PLR0915
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert self.space.fidelity is None, "Fidelity not supported yet."
        parameters = self.space.searchables

        n_to_sample = 1 if n is None else n
        n_sampled = len(trials)
        id_generator = iter(str(i) for i in itertools.count(n_sampled + 1))

        # If the amount of configs evaluated is less than the initial design
        # requirement, keep drawing from initial design
        n_evaluated = sum(
            1
            for trial in trials.values()
            if trial.report is not None and trial.report.objective_to_minimize is not None
        )
        sampled_configs: list[SampledConfig] = []

        if n_evaluated < self.n_initial_design:
            # For reproducibility, we need to ensure we do the same sample of all
            # configs each time.
            design_samples = make_initial_design(
                parameters=parameters,
                encoder=self.encoder,
                sample_prior_first=self.sample_prior_first if n_sampled == 0 else False,
                sampler=self.prior if self.prior is not None else "uniform",
                seed=None,  # TODO: Seeding, however we need to avoid repeating configs
                sample_size=self.n_initial_design,
            )

            # Then take the subset we actually need
            design_samples = design_samples[n_evaluated:]
            for sample in design_samples:
                sample.update(self.space.constants)

            sampled_configs.extend(
                [
                    SampledConfig(id=config_id, config=config)
                    for config_id, config in zip(
                        id_generator,
                        design_samples,
                        # NOTE: We use a generator for the ids so no need for strict
                        strict=False,
                    )
                ]
            )

            if len(sampled_configs) >= n_to_sample:
                return sampled_configs[0] if n is None else sampled_configs

        # Otherwise, we encode trials and setup to fit and acquire from a GP
        data, encoder = encode_trials_for_gp(
            trials,
            parameters,
            device=self.device,
            encoder=self.encoder,
        )

        cost_percent = None
        if self.cost_aware:
            # TODO: Interaction with `"log"` cost aware
            if self.cost_aware == "log":
                raise NotImplementedError("Log cost aware not implemented yet.")

            if budget_info is None:
                raise ValueError(
                    "Must provide a 'cost' to configurations if using cost"
                    " with BayesianOptimization."
                )
            if budget_info.max_cost_total is None:
                raise ValueError("Cost budget must be set if using cost")
            cost_percent = budget_info.used_cost_budget / budget_info.max_cost_total

        # If we should use the prior, weight the acquisition function by
        # the probability of it being sampled from the prior.
        pibo_exp_term = None
        prior = None
        if self.prior:
            pibo_exp_term = _pibo_exp_term(n_sampled, encoder.ndim, self.n_initial_design)
            # If the exp term is insignificant, skip prior acq. weighting
            prior = None if pibo_exp_term < 1e-4 else self.prior

        n_to_acquire = n_to_sample - len(sampled_configs)

        num_objectives = None
        for trial in trials.values():
            if trial.report is not None:
                match trial.report.objective_to_minimize:
                    case None:
                        continue
                    case Sequence():
                        if num_objectives is None:
                            num_objectives = len(trial.report.objective_to_minimize)
                        assert (
                            len(trial.report.objective_to_minimize) == num_objectives
                        ), "All trials must have the same number of objectives."
                    case float():
                        if num_objectives is None:
                            num_objectives = 1
                        assert num_objectives == 1, (
                            "All trials must have the same number of objectives."
                        )
                    case _:
                        raise TypeError(
                            "Objective to minimize must be a float or a sequence "
                            "of floats."
                        )

        # Sanity check, but this shouldn't happen in the first place since we
        # check `n_evaluated < self.n_initial_design` above.
        assert num_objectives is not None, (
            "Either no trials have been completed or"
            " No trials reports have objective values."
        )

        if num_objectives > 1:
            gp = make_default_single_obj_gp(
                x=data.x,
                y=data.y,
                encoder=encoder,
                y_transform=Standardize(m=data.y.shape[-1]),
            )
            if self.reference_point is not None:
                assert len(self.reference_point) == num_objectives, (
                    "Reference point must have the same number of objectives as the"
                    " trials."
                )
                ref_point = torch.tensor(
                    self.reference_point,
                    dtype=data.x.dtype,
                    device=data.x.device,
                )
            else:
                ref_point = torch.tensor(
                    _get_reference_point(data.y.numpy()),
                    dtype=data.x.dtype,
                    device=data.x.device,
                )
            acquisition = qLogNoisyExpectedHypervolumeImprovement(
                model=gp,
                ref_point=ref_point,
                X_baseline=data.x,
                X_pending=data.x_pending,
                prune_baseline=True,
            )
        else:
            gp = make_default_single_obj_gp(x=data.x, y=data.y, encoder=encoder)
            acquisition = qLogNoisyExpectedImprovement(
                model=gp,
                X_baseline=data.x,
                # Unfortunatly, there's no option to indicate that we minimize
                # the AcqFunction so we need to do some kind of transformation.
                # https://github.com/pytorch/botorch/issues/2316#issuecomment-2085964607
                objective=LinearMCObjective(weights=torch.tensor([-1.0])),
                X_pending=data.x_pending,
                prune_baseline=True,
            )
        candidates = fit_and_acquire_from_gp(
            gp=gp,
            x_train=data.x,
            encoder=encoder,
            acquisition=acquisition,
            prior=prior,
            n_candidates_required=n_to_acquire,
            pibo_exp_term=pibo_exp_term,
            costs=data.cost if self.cost_aware is not False else None,
            cost_percentage_used=cost_percent,
            costs_on_log_scale=self.cost_aware == "log",
            hide_warnings=True,
        )

        configs = encoder.decode(candidates)
        for config in configs:
            config.update(self.space.constants)

        sampled_configs.extend(
            [
                SampledConfig(id=config_id, config=config)
                for config_id, config in zip(
                    id_generator,
                    configs,
                    # NOTE: We use a generator for the ids so no need for strict
                    strict=False,
                )
            ]
        )
        return sampled_configs[0] if n is None else sampled_configs


def _get_reference_point(loss_vals: np.ndarray) -> np.ndarray:
    """Get the reference point from the completed Trials."""
    eps = 1e-4
    worst_point = np.max(loss_vals, axis=0)
    return worst_point * (1 + eps)
