from functools import partial
from typing import Any, Mapping, Literal

import numpy as np
import torch

import warnings
from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.bayesian_optimization.models.ftpfn import (
    FTPFNSurrogate,
    acquire_next_from_ftpfn,
    encode_trials_for_ftpfn,
)
from neps.optimizers.intial_design import make_initial_design
from neps.sampling.priors import Prior
from neps.sampling.samplers import Sampler
from neps.search_spaces.domain import Domain
from neps.search_spaces.encoding import CategoricalToUnitNorm, ConfigEncoder
from neps.search_spaces.search_space import FloatParameter, IntegerParameter, SearchSpace
from neps.state.trial import Trial
from neps.state.optimizer import BudgetInfo


# NOTE: Ifbo was trained using 32 bit
FTPFN_DTYPE = torch.float32
ID_COL = 0
BUDGET_COL = 1


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
    assert isinstance(fidelity, FloatParameter | IntegerParameter)
    if fidelity.log:
        raise NotImplementedError("Log fidelity not yet supported")

    # Can't use mod since it's quite innacurate for floats
    # Use the fact that we can always write x = n*k + r
    # where k = stepsize and x = (fid_upper - fid_lower)
    # x = n*k + r
    # n = x // k
    # r = x - n*k
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
        default=fidelity.default,
        is_fidelity=True,
        default_confidence=fidelity.default_confidence_choice,
    )
    return (
        SearchSpace(**{**pipeline_space.hyperparameters, fidelity_name: new_fid}),
        n,
    )


def _tokenize(
    ids: torch.Tensor,
    budgets: torch.Tensor,
    configs: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([ids.unsqueeze(1), budgets.unsqueeze(1), configs], dim=1).to(
        FTPFN_DTYPE
    )


def _encode_for_ftpfn(
    trials: Mapping[str, Trial],
    encoder: ConfigEncoder,
    space: SearchSpace,
    budget_domain: Domain,
    device: torch.device | None = None,
    dtype: torch.dtype = FTPFN_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the trials into a format that the FTPFN model can understand.

    !!! warning "loss values reported"

        The `ys` are a single dimension but consist of the losses inverted to scores.
        As result, we have to assert that the loss values provided in the trials are
        in the range [0, 1].

    !!! note "X layout"

        The layout of the X is:

        ```
        | config_id | budget (normalized from fidelity) | hp_1 | hp_2 | ... | hp_n |
        ```

        Here the `budget` is normalized to the range [0, 1] while the hp parameters
        are encoded according to the provided encoder, which should map the parameter
        values from the original domain to some domain in [0, 1].

    !!! warning "Pending and Error trials"

        We currently do not handle error cases, **and they are ignored**.
        For trials which do not have a loss reported yet, they are considered pending
        and will have `torch.nan` as their score inside the returned y values.

    Args:
        trials: The trials to encode
        encoder: The encoder to use
        space: The search space
        budget_domain: The domain to use for the budgets of the FTPFN
        device: The device to use
        dtype: The dtype to use

    Returns:
        The encoded trials and their corresponding **scores**
    """
    # Select all trials which have something we can actually use for modelling
    # The absence of a report signifies pending
    selected = {
        trial_id: trial
        for trial_id, trial in trials.items()
        if trial.report is None or trial.report.loss is not None
    }
    assert space.fidelity_name is not None
    assert space.fidelity is not None
    train_configs = encoder.encode([t.config for t in selected.values()], device=device)
    ids = torch.tensor(
        [int(config_id.split("_", maxsplit=1)[0]) for config_id in selected.keys()],
        device=device,
        dtype=torch.float64,
    )
    train_fidelities = torch.tensor(
        [t.config[space.fidelity_name] for t in selected.values()],
        device=device,
        dtype=torch.float64,
    )
    train_budgets = budget_domain.cast(train_fidelities, frm=space.fidelity.domain)
    X = _tokenize(
        ids=torch.tensor(ids, device=device),
        budgets=train_budgets,
        configs=train_configs,
    ).to(dtype)

    # TODO: Document that it's on the user to ensure these are already all bounded
    # We could possibly include some bounded transform to assert this.
    minimize_ys = torch.tensor(
        [
            trial.report.loss
            if trial.report is not None and trial.report.loss is not None
            else np.nan
            for trial in trials.values()
        ],
        device=device,
        dtype=FTPFN_DTYPE,
    )
    if minimize_ys.max() > 1 or minimize_ys.min() < 0:
        raise RuntimeError(
            "ifBO requires that all loss values reported lie in the interval [0, 1]"
            " but recieved loss value outside of that range!"
            f"\n{minimize_ys}"
        )
    maximize_ys = 1 - minimize_ys
    return X, maximize_ys


def _keep_highest_budget_evaluation(x: torch.Tensor) -> torch.Tensor:
    # Does a lexsort, same as if we sorted by (config_id, budget), where
    # theyre are sorted according to increasing config_id and then increasing budget.
    # x[i2] -> sorted by config id and budget
    i1 = torch.argsort(x[:, BUDGET_COL])
    i2 = i1[torch.argsort(x[i1][:, ID_COL], stable=True)]
    sorted_x = x[i2]

    # Now that it's sorted, we essentially want to count the occurence of each id into counts
    _, counts = torch.unique_consecutive(sorted_x[:, ID_COL], return_counts=True)

    # Now we can use these counts to get to the last occurence of each id
    # The -1 is because we want to index from 0 but sum starts at 1.
    ii = counts.cumsum(0) - 1
    return sorted_x[ii]


class IFBO(BaseOptimizer):
    """Base class for MF-BO algorithms that use DyHPO-like acquisition and budgeting."""

    def __init__(
        self,
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
        loss_value_on_error: float | None = None,  # TODO: Remove
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
            initial_design_size: Number of configurations to sample before starting optimization

                If None, the number of configurations will be equal to the number of dimensions.

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

        params = {**space.numerical, **space.categoricals}
        self._prior = Prior.from_parameters(params) if use_priors else None
        self._config_encoder: ConfigEncoder = ConfigEncoder.default(
            params,
            # FTPFN doesn't support categoricals and we were recomenned to just evenly distribute
            # in the unit norm
            custom_transformers={
                cat_name: CategoricalToUnitNorm(choices=cat.choices)
                for cat_name, cat in space.categoricals.items()
            },
        )
        self._border_sampler = Sampler.borders(len(params))
        self._cached_border_configs: torch.Tensor | None = None

        # Domain of fidelity values, i.e. what is given in the configs that we
        # give to the user to evaluate at.
        self._fid_domain = space.fidelity.domain

        # Domain in which we should pass budgets to ifbo model
        self._budget_domain = Domain.float(1 / self._max_budget, 1)

        # Domain from which we assign an index to each budget
        self._budget_ix_domain = Domain.indices(fid_bins)

    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo,
        optimizer_state: dict[str, Any],
        seed: torch.Generator | None = None,
    ) -> SampledConfig:
        if seed is not None:
            raise NotImplementedError("Seed is not yet implemented for IFBO")

        ids = [int(config_id.split("_", maxsplit=1)[0]) for config_id in trials.keys()]
        new_id = max(ids) + 1 if len(ids) > 0 else 0

        # If we havn't passed the intial design phase
        if self._initial_design is None:
            self._initial_design = make_initial_design(
                space=self.pipeline_space,
                encoder=self._config_encoder,
                sample_default_first=self.sample_default_first,
                sampler="sobol" if self._prior is None else self._prior,
                seed=seed,
                sample_fidelity="min",
                sample_size=self.n_initial_design,
            )

        if new_id < len(self._initial_design):
            return SampledConfig(id=f"{new_id}_0", config=self._initial_design[new_id])

        # Otherwise, we proceed to surrogate phase
        data = encode_trials_for_ftpfn(
            trials=trials,
            space=self.pipeline_space,
            encoder=self._config_encoder,
            budget_domain=self._budget_domain,
            device=self.device,
        )

        # TODO: Very little chance mfpi_random is best but for now it's stable
        def _mfpi_random(
            _X: torch.Tensor,
            _y: torch.Tensor,
            _acq_samples: torch.Tensor,
            _ftpfn: FTPFNSurrogate,
            how: Literal["pi", "ei"],
        ) -> torch.Tensor:
            rng = np.random.RandomState(None if seed is None else seed + len(trials))
            _low = self._budget_ix_domain.lower
            _high = self._budget_ix_domain.upper
            horizon_index = rng.randint(_low, _high) + 1
            horizon = self._budget_domain.cast_one(
                horizon_index, frm=self._budget_ix_domain
            )
            f_best = _y.max().item()
            r = rng.uniform(-4, -1)
            threshold = f_best + (10**r) * (1 - f_best)

            # NOTE: If converting f_inc to be seperate per acq sample, you
            # need to add an extra batch dimension to y_best, i.e. (n, 1)
            # Budget column is between 0 and 1, but we want to add the horizon
            BUDGET_COL = 1
            _acq_samples[:, BUDGET_COL] += horizon
            _acq_samples[:, BUDGET_COL] = torch.clamp(
                _acq_samples[:, BUDGET_COL], max=self._budget_domain.upper
            )

            match how:
                case "pi":
                    return _ftpfn.get_pi(_X, _y, _acq_samples, y_best=threshold)
                case "ei":
                    return _ftpfn.get_ei(_X, _y, _acq_samples, y_best=threshold)
                case _:
                    raise ValueError(f"Unknown acquisition strategy: {how=}")

        ndims = self._config_encoder.ncols

        # Sample some configurations at uniform for acq.
        uniform_sampler = Sampler.uniform(ndim=ndims)
        uniform_configs = uniform_sampler.sample(
            self.n_acquisition_new_configs,
            to=self._config_encoder.domains,
            seed=seed,
            device=self.device,
            dtype=FTPFN_DTYPE,
        )

        # Also sample some border configurations for acq.
        # OPTIM: If we are below the amount possible, there is no randomness and we can cache them
        border_sampler = Sampler.borders(ndim=ndims)
        N_border = 2**9  # 512, if we go over, we subselect 512 border configs
        if N_border <= border_sampler.n_possible:
            if self._cached_border_configs is not None:
                border_configs = self._cached_border_configs
            else:
                self._cached_border_configs = border_sampler.sample(
                    n=N_border,
                    to=self._config_encoder.domains,
                    seed=seed,
                    device=self.device,
                    dtype=FTPFN_DTYPE,
                )
                border_configs = self._cached_border_configs
        else:
            border_configs = border_sampler.sample(
                n=N_border,
                to=self._config_encoder.domains,
                seed=seed,
                device=self.device,
                dtype=FTPFN_DTYPE,
            )

        id, current_fid, config = acquire_next_from_ftpfn(
            ftpfn=FTPFNSurrogate(
                target_path=self.surrogate_model_args.get("target_path", None),
                version=self.surrogate_model_args.get("version", "0.0.1"),
                device=self.device,
            ),
            data=data,
            seed=seed,
            encoder=self._config_encoder,
            budget_domain=self._budget_domain,
            fidelity_domain=self._fid_domain,
            extra_acq_samples=torch.cat([uniform_configs, border_configs], dim=0),
            acq_strategy=partial(_mfpi_random, how="ei"),
        )
        if current_fid is None:
            assert id is None
            config[self._fidelity_name] = self._fid_domain.lower
            return SampledConfig(id=f"{new_id}_0", config=config)
        else:
            current_budget_ix = self._budget_ix_domain.cast_one(
                current_fid, frm=self._fid_domain
            )
            next_budget_ix = current_budget_ix + 1
            next_fid = self._fid_domain.cast_one(
                next_budget_ix, frm=self._budget_ix_domain
            )
            config[self._fidelity_name] = next_fid
            return SampledConfig(
                id=f"{id}_{next_budget_ix}",
                config=config,
                previous_config_id=f"{id}_{current_budget_ix}",
            )
