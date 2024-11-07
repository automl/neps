from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import numpy as np
import torch

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.bayesian_optimization.models.ftpfn import (
    FTPFNSurrogate,
    acquire_next_from_ftpfn,
    decode_ftpfn_data,
    encode_ftpfn,
)
from neps.optimizers.initial_design import make_initial_design
from neps.sampling.priors import Prior
from neps.sampling.samplers import Sampler
from neps.search_spaces.domain import Domain
from neps.search_spaces.encoding import CategoricalToUnitNorm, ConfigEncoder
from neps.search_spaces.search_space import Float, Integer, SearchSpace

if TYPE_CHECKING:
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial

# NOTE: Ifbo was trained using 32 bit
FTPFN_DTYPE = torch.float32


def _adjust_pipeline_space_to_match_stepsize(
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
    # > x = n*k + r
    # > n = x // k
    # > r = x - n*k
    x = fidelity.upper - fidelity.lower
    n = int(x // step_size)

    if n <= 0:
        raise ValueError(
            f"Step size ({step_size}) is too large for the fidelity domain {fidelity}."
            "Considering lowering this parameter to ifBO."
        )

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


class IFBO(BaseOptimizer):
    """Base class for MF-BO algorithms that use DyHPO-like acquisition and budgeting."""

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        step_size: int | float = 1,
        use_priors: bool = False,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
        surrogate_model_args: dict | None = None,
        initial_design_size: int | Literal["ndim"] = "ndim",
        n_acquisition_new_configs: int = 1_000,
        device: torch.device | None = None,
        budget: int | float | None = None,  # TODO: Remove
        objective_to_minimize_value_on_error: float | None = None,  # TODO: Remove
        cost_value_on_error: float | None = None,  # TODO: Remove
        ignore_errors: bool = False,  # TODO: Remove
    ):
        """Initialise.

        Args:
            pipeline_space: Space in which to search
            step_size: The size of the step to take in the fidelity domain.
            sampling_policy: The type of sampling procedure to use
            promotion_policy: The type of promotion procedure to use
            sample_default_first: Whether to sample the default configuration first
            initial_design_size: Number of configs to sample before starting optimization

                If None, the number of configs will be equal to the number of dimensions.

            device: Device to use for the model
        """
        # TODO: I'm not sure how this might effect tables, whose lowest fidelity
        # might be below to possibly increased lower bound.
        space, fid_bins = _adjust_pipeline_space_to_match_stepsize(
            pipeline_space, step_size
        )
        assert space.fidelity is not None
        assert isinstance(space.fidelity_name, str)

        super().__init__(pipeline_space=space)
        self.step_size = step_size
        self.use_priors = use_priors
        self.sample_default_first = sample_default_first
        self.sample_default_at_target = sample_default_at_target
        self.device = device
        self.n_initial_design: int | Literal["ndim"] = initial_design_size
        self.n_acquisition_new_configs = n_acquisition_new_configs
        self.surrogate_model_args = surrogate_model_args or {}

        self._min_budget: int | float = space.fidelity.lower
        self._max_budget: int | float = space.fidelity.upper
        self._fidelity_name: str = space.fidelity_name
        self._initial_design: list[dict[str, Any]] | None = None

        self._prior: Prior | None
        if use_priors:
            self._prior = Prior.from_space(space, include_fidelity=False)
        else:
            self._prior = None

        self._config_encoder: ConfigEncoder = ConfigEncoder.from_space(
            space=space,
            include_constants_when_decoding=True,
            # FTPFN doesn't support categoricals and we were recomended
            # to just evenly distribute in the unit norm
            custom_transformers={
                cat_name: CategoricalToUnitNorm(choices=cat.choices)
                for cat_name, cat in space.categoricals.items()
            },
        )

        # Domain of fidelity values, i.e. what is given in the configs that we
        # give to the user to evaluate at.
        self._fid_domain = space.fidelity.domain

        # Domain in which we should pass budgets to ifbo model
        self._budget_domain = Domain.floating(1 / self._max_budget, 1)

        # Domain from which we assign an index to each budget
        self._budget_ix_domain = Domain.indices(fid_bins)

    @override
    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
    ) -> SampledConfig:
        ids = [int(config_id.split("_", maxsplit=1)[0]) for config_id in trials]
        new_id = max(ids) + 1 if len(ids) > 0 else 0

        # If we havn't passed the intial design phase
        if self._initial_design is None:
            self._initial_design = make_initial_design(
                space=self.pipeline_space,
                encoder=self._config_encoder,
                sample_default_first=self.sample_default_first,
                sampler="sobol" if self._prior is None else self._prior,
                seed=None,  # TODO:
                sample_fidelity="min",
                sample_size=self.n_initial_design,
            )

        if new_id < len(self._initial_design):
            config = self._initial_design[new_id]
            config[self._fidelity_name] = self._min_budget
            return SampledConfig(id=f"{new_id}_0", config=config)

        # Otherwise, we proceed to surrogate phase
        ftpfn = FTPFNSurrogate(
            target_path=self.surrogate_model_args.get("target_path", None),
            version=self.surrogate_model_args.get("version", "0.0.1"),
            device=self.device,
        )
        X, y = encode_ftpfn(
            trials=trials,
            space=self.pipeline_space,
            encoder=self._config_encoder,
            budget_domain=self._budget_domain,
            device=self.device,
            pending_value=torch.nan,
        )

        # Fantasize if needed
        pending_mask = torch.isnan(y)
        if pending_mask.any():
            not_pending_mask = ~pending_mask
            not_pending_X = X[not_pending_mask]
            y[pending_mask] = ftpfn.get_mean_performance(
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
        horizon_increment = self._budget_domain.cast_one(
            rng.randint(*self._budget_ix_domain.bounds) + 1,
            frm=self._budget_ix_domain,
        )
        f_best = y.max().item()
        threshold = f_best + (10 ** rng.uniform(-4, -1)) * (1 - f_best)

        def _mfpi_random(samples: torch.Tensor) -> torch.Tensor:
            # HACK: Because we are modifying the samples inplace, we do,
            # and then undo the addition
            original_budget_column = samples[..., 1].clone()
            samples[..., 1].add_(horizon_increment).clamp_max_(self._budget_domain.upper)

            scores = ftpfn.get_pi(X, y, samples, y_best=threshold)

            samples[..., 1] = original_budget_column
            return scores

        # Do acquisition on ftpfn
        sample_dims = self._config_encoder.ncols
        best_row = acquire_next_from_ftpfn(
            ftpfn=ftpfn,
            # How to encode
            encoder=self._config_encoder,
            budget_domain=self._budget_domain,
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
            self._config_encoder,
            budget_domain=self._budget_domain,
            fidelity_domain=self._fid_domain,
        )[0]

        if _id is None:
            config[self._fidelity_name] = fid
            return SampledConfig(id=f"{new_id}_0", config=config)
        # Convert fidelity to budget index, bump by 1 and convert back
        budget_ix = self._budget_ix_domain.cast_one(fid, frm=self._fid_domain)
        next_ix = budget_ix + 1
        next_fid = self._fid_domain.cast_one(next_ix, frm=self._budget_ix_domain)

        config[self._fidelity_name] = next_fid
        return SampledConfig(
            id=f"{_id}_{next_ix}",
            config=config,
            previous_config_id=f"{_id}_{budget_ix}",
        )
