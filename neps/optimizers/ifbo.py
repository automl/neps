from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from neps.optimizers.models.ftpfn import (
    FTPFNSurrogate,
    acquire_next_from_ftpfn,
    decode_ftpfn_data,
    encode_ftpfn,
)
from neps.optimizers.optimizer import SampledConfig
from neps.optimizers.utils.initial_design import make_initial_design
from neps.sampling.samplers import Sampler
from neps.search_spaces.domain import Domain
from neps.search_spaces.search_space import Float, Integer, SearchSpace

if TYPE_CHECKING:
    from neps.sampling.priors import Prior
    from neps.search_spaces.encoding import ConfigEncoder
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial

# NOTE: Ifbo was trained using 32 bit
FTPFN_DTYPE = torch.float32


def adjust_pipeline_space_to_match_stepsize(
    pipeline_space: SearchSpace,
    step_size: int | float,
) -> tuple[SearchSpace, int]:
    """Adjust the pipeline space to be evenly divisible by the step size.

    This is done by incrementing the lower bound of the fidelity domain to the
    that enables this.

    Args:
        pipeline_space: The pipeline space to adjust
        step_size: The size of the step to take in the fidelity domain.

    Returns:
        The adjusted pipeline space and the number of bins it can be divided into
    """
    fidelity = pipeline_space.fidelity
    fidelity_name = pipeline_space.fidelity_name
    assert fidelity_name is not None
    assert isinstance(fidelity, Float | Integer)
    if fidelity.log:
        raise NotImplementedError("Log fidelity not yet supported")

    # Can't use mod since it's quite innacurate for floats
    # Use the fact that we can always write x = n*k + r
    # where k = stepsize and x = (fid_upper - fid_lower)

    x = fidelity.upper - fidelity.lower

    # > x = n*k + r
    # > n = x // k
    n = int(x // step_size)

    if n <= 0:
        raise ValueError(
            f"Step size ({step_size}) is too large for the fidelity domain {fidelity}."
            "Considering lowering this parameter to ifBO."
        )

    # > r = x - n*k
    r = x - n * step_size
    new_lower = fidelity.lower + r
    new_fid = fidelity.__class__(
        lower=new_lower,
        upper=fidelity.upper,
        log=fidelity.log,
        prior=fidelity.prior,
        is_fidelity=True,
        prior_confidence=fidelity.prior_confidence_choice,
    )
    return (
        SearchSpace(**{**pipeline_space.hyperparameters, fidelity_name: new_fid}),
        n,
    )


@dataclass
class IFBO:
    pipeline_space: SearchSpace
    encoder: ConfigEncoder
    sample_prior_first: bool
    prior: Prior | None
    n_initial_design: int

    device: torch.device | None
    ftpfn: FTPFNSurrogate

    fid_domain: Domain
    fidelity_name: str
    n_fidelity_bins: int

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "TODO"
        ids = [int(config_id.split("_", maxsplit=1)[0]) for config_id in trials]
        new_id = max(ids) + 1 if len(ids) > 0 else 0

        min_fid = self.fid_domain.lower
        max_fid = self.fid_domain.upper

        # The FTPFN surrogate takes in a budget in the range [0, 1]
        # We also need to be able to map these to discrete integers
        # Hence we use the two domains below to do so.

        # Domain in which we should pass budgets to ifbo model
        budget_domain = Domain.floating(lower=1 / max_fid, upper=1)

        # Domain from which we assign an index to each budget
        budget_index_domain = Domain.indices(self.n_fidelity_bins)

        # If we havn't passed the intial design phase
        if self._initial_design is None:
            self._initial_design = make_initial_design(
                space=self.pipeline_space,
                encoder=self.encoder,
                sample_prior_first=self.sample_prior_first,
                sampler="sobol" if self.prior is None else self.prior,
                seed=None,  # TODO:
                sample_fidelity="min",
                sample_size=self.n_initial_design,
            )

        if new_id < len(self._initial_design):
            config = self._initial_design[new_id]
            config[self.fidelity_name] = min_fid
            return SampledConfig(id=f"{new_id}_0", config=config)

        X, y = encode_ftpfn(
            trials=trials,
            space=self.pipeline_space,
            encoder=self.encoder,
            budget_domain=budget_domain,
            device=self.device,
            pending_value=torch.nan,
        )

        # Fantasize if needed
        pending_mask = torch.isnan(y)
        if pending_mask.any():
            not_pending_mask = ~pending_mask
            not_pending_X = X[not_pending_mask]
            y[pending_mask] = self.ftpfn.get_mean_performance(
                train_x=not_pending_X,
                train_y=y[not_pending_mask],
                test_x=X[pending_mask],
            )
        else:
            not_pending_X = X

        # NOTE: Can't really abstract this, requires knowledge that:
        # 1. The encoding is such that the objective_to_minimize is 1 -
        # objective_to_minimize
        # 2. The budget is the second column
        # 3. The budget is encoded between 1/max_fid and 1
        rng = np.random.RandomState(len(trials))
        # Cast the a random budget index into the ftpfn budget domain
        horizon_increment = budget_domain.cast_one(
            rng.randint(*budget_index_domain.bounds) + 1,
            frm=budget_index_domain,
        )
        f_best = y.max().item()
        threshold = f_best + (10 ** rng.uniform(-4, -1)) * (1 - f_best)

        def _mfpi_random(samples: torch.Tensor) -> torch.Tensor:
            # HACK: Because we are modifying the samples inplace, we do,
            # and then undo the addition
            original_budget_column = samples[..., 1].clone()
            samples[..., 1].add_(horizon_increment).clamp_max_(budget_domain.upper)

            scores = self.ftpfn.get_pi(X, y, samples, y_best=threshold)

            samples[..., 1] = original_budget_column
            return scores

        # Do acquisition on ftpfn
        # TODO: Parametrize some of this
        sample_dims = self.encoder.ndim
        best_row = acquire_next_from_ftpfn(
            ftpfn=self.ftpfn,
            # How to encode
            encoder=self.encoder,
            budget_domain=budget_domain,
            # Acquisition function
            acq_function=_mfpi_random,
            # Which acquisition samples to consider for continuation
            continuation_samples=not_pending_X,
            # How to generate some initial samples
            initial_samplers=[
                (Sampler.sobol(ndim=sample_dims), 512),
                (Sampler.uniform(ndim=sample_dims), 512),
                (Sampler.borders(ndim=sample_dims), 256),
            ],
            seed=None,  # TODO: Seeding
            # A next step local sampling around best point found by initial_samplers
            local_search_sample_size=256,
            local_search_confidence=0.95,
        )
        _id, fid, config = decode_ftpfn_data(
            best_row,
            self.encoder,
            budget_domain=budget_domain,
            fidelity_domain=self.fid_domain,
        )[0]

        if _id is None:
            config[self.fidelity_name] = fid
            return SampledConfig(id=f"{new_id}_0", config=config)

        # Convert fidelity to budget index, bump by 1 and convert back
        budget_ix = budget_index_domain.cast_one(fid, frm=self.fid_domain)
        next_ix = budget_ix + 1
        next_fid = self.fid_domain.cast_one(next_ix, frm=budget_index_domain)

        config[self.fidelity_name] = next_fid
        return SampledConfig(
            id=f"{_id}_{next_ix}",
            config=config,
            previous_config_id=f"{_id}_{budget_ix}",
        )
