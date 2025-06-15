from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from gpytorch.utils.warnings import NumericalWarning

from neps.optimizers.models.gp import (
    encode_trials_for_gp,
    fit_and_acquire_from_gp,
    make_default_single_obj_gp,
)
from neps.optimizers.optimizer import SampledConfig
from neps.utils.common import disable_warnings

if TYPE_CHECKING:
    from neps.optimizers.bracket_optimizer import BracketOptimizer
    from neps.sampling.priors import Prior
    from neps.space import SearchSpace
    from neps.space.encoding import ConfigEncoder
    from neps.state import BudgetInfo, Trial


@dataclass
class PriMO:
    """The PriMO algorithm for integrating MO priors in Bayesian Optimization."""

    space: SearchSpace
    """The search space to use, without the fidelity."""

    encoder: ConfigEncoder
    """The encoder to use for the search space."""

    bracket_optimizer: BracketOptimizer
    """The bracket optimizer to use for the initial design."""

    initial_design_size: int
    """The number of initial designs to use."""

    fid_max: int | float
    """The maximum fidelity value in the BracketOptimizer's search space."""

    fid_name: str
    """The name of the fidelity in the BracketOptimizer's search space."""

    scalarization_weights: dict[str, float] | None = None
    """The scalarization weights to use for the objectives for BO."""

    device: torch.device | None = None
    """The device to use for the GP optimization."""

    priors: Mapping[str, Prior] | None = None
    """The priors to use for this optimizer."""

    n_init_used: int = field(default=0, init=False)
    """The effective number of initial seed configurations used
    for the Bayesian optimization. This refers to the number of
    configurations that were evaluated at the maximum fidelity.
    """

    epsilon: float = 0.25
    """The epsilon value to use for the epsilon-greedy decaying prior-weighted
    acquisition function. This is the probability of not using the prior
    acquisition function.
    """

    def __call__(  # noqa: C901, PLR0912
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "TODO"

        # Use MOASHA in the initial design phase, i.e., until the threshold is reached
        if not self.threshold_reached(
            trials=trials,
            threshold=self.initial_design_size,
        ):
            return self.sample_using_initial_design(
                trials=trials,
                budget_info=budget_info,
                n=n,
            )

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

        # Set scalarization weights if not set
        if self.scalarization_weights is None:
            self.scalarization_weights = np.random.uniform(size=num_objectives)
            self.scalarization_weights /= np.sum(self.scalarization_weights)

        # Scalarize trials.report.objective_to_minimize and remove fidelity
        # from the trial configs
        nxt_id = 1
        _trials = {}
        for trial_id, trial in trials.items():
            _trial = copy.deepcopy(trial)
            # Skip trials that are not evaluated at the maximum fidelity
            if _trial.config.get(self.fid_name) != self.fid_max:
                continue
            # Skip trials that are still pending
            if _trial.report is None:
                continue
            assert _trial.report.objective_to_minimize is not None, (
                f"Trial {trial_id} has no objective to minimize."
            )
            assert isinstance(_trial.report.objective_to_minimize, Sequence), (
                "Trial objectives must be a sequence for PriMO, "
                f"got {type(_trial.report.objective_to_minimize)}"
            )

            # Convert the objective to an ndarray and scalarize it
            if not isinstance(_trial.report.objective_to_minimize, np.ndarray):
                _trial.report.objective_to_minimize = np.array(
                    _trial.report.objective_to_minimize
                )
            _trial.report.objective_to_minimize = np.dot(
                _trial.report.objective_to_minimize, self.scalarization_weights
            )

            # Remove the fidelity from the trial config
            # Cannot do simple pop since Config type is Mapping in most places in Neps
            _trial.config = {k: v for k, v in _trial.config.items() if k != self.fid_name}

            _trials[trial_id] = _trial

            # Get the next ID for the sampled configuration
            if "_" in trial_id:
                config_id_str, _ = trial_id.split("_")
            else:
                config_id_str = trial_id

            nxt_id = max(nxt_id, int(config_id_str) + 1)

        assert len(_trials) > 0, (
            "No trials found with the maximum fidelity. "
            "Consider increasing the initial design size to run MOASHA longer."
        )

        if self.n_init_used == 0:
            self.n_init_used = len(_trials)

        # Sample new configurations using the Bayesian optimization
        sampled_config = self.sample_using_bo(
            trials=_trials,
            budget_info=budget_info,
            n=n,
        )
        sampled_config.update(
            {
                self.fid_name: self.fid_max,
                **self.space.constants,
            }
        )
        return SampledConfig(id=str(nxt_id), config=sampled_config)

    def sample_using_initial_design(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        """Use the initial design to sample new configurations."""
        assert n is None, "TODO"

        # Sample new configurations using the initial design
        return self.bracket_optimizer(
            trials=trials,
            budget_info=budget_info,
            n=n,
        )

    def sample_using_bo(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> dict[str, Any]:
        """Use Bayesian optimization to sample new configurations."""
        assert n is None, "TODO"
        n_sampled = len(trials)

        data, encoder = encode_trials_for_gp(
            trials,
            self.space.searchables,
            device=self.device,
            encoder=self.encoder,
        )

        selected_prior = None
        if self.priors is not None:
            selected_prior = np.random.choice(
                list(self.priors.values()),
            )

        selected_prior = np.random.choice(
            [selected_prior, None],
            p=[1 - self.epsilon, self.epsilon],
        )

        # If we should use the prior, weight the acquisition function by
        # the probability of it being sampled from the prior.
        pibo_exp_term = None
        prior = None
        if selected_prior:
            pibo_exp_term = _pibo_exp_term(n_sampled, encoder.ndim, self.n_init_used)
            # If the exp term is insignificant, skip prior acq. weighting
            prior = None if pibo_exp_term < 1e-4 else selected_prior

        n_to_acquire = 1

        gp = make_default_single_obj_gp(x=data.x, y=data.y, encoder=encoder)
        with disable_warnings(NumericalWarning):
            acquisition = qLogNoisyExpectedImprovement(
                model=gp,
                X_baseline=data.x,
                # Unfortunatly, there's no option to indicate that we minimize
                # the AcqFunction so we need to do some kind of transformation.
                # https://github.com/pytorch/botorch/issues/2316#issuecomment-2085964607
                objective=LinearMCObjective(
                    weights=torch.tensor([-1.0], device=self.device)
                ),
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
            hide_warnings=True,
        )

        return encoder.decode_one(candidates)

    def threshold_reached(
        self,
        trials: Mapping[str, Trial],
        threshold: int | float,
    ) -> bool:
        used_fidelity = [
            t.config[self.fid_name] for t in trials.values() if t.report is not None
        ]
        fidelity_units_used = sum(used_fidelity) / self.fid_max
        return fidelity_units_used >= threshold


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
    import math

    n_bo_samples = n_sampled_already - initial_design_size
    return math.exp(-(n_bo_samples**2) / ndims)
