from typing import Any, Mapping, Literal

import math
import numpy as np
import torch

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.bayesian_optimization.models.ftpfn import FTPFNSurrogate
from neps.optimizers.intial_design import make_initial_design
from neps.sampling.priors import Prior
from neps.sampling.samplers import Sampler
from neps.search_spaces.domain import Domain
from neps.search_spaces.encoding import CategoricalToUnitNorm, TensorEncoder
from neps.search_spaces.search_space import FloatParameter, IntegerParameter, SearchSpace
from neps.state.trial import Trial
from neps.state.optimizer import BudgetInfo


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
    return torch.cat([ids.unsqueeze(1), budgets.unsqueeze(1), configs], dim=1)


def _encode_for_ftpfn(
    trials: Mapping[str, Trial],
    encoder: TensorEncoder,
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
    # TODO: Currently we do not handle error cases, we can't use NaN as that
    # is what we use for trials that have no loss yet, i.e. pending trials.
    selected = {
        trial_id: trial
        for trial_id, trial in trials.items()
        if trial.state
        not in (
            Trial.State.FAILED,
            Trial.State.CRASHED,
            Trial.State.UNKNOWN,
            Trial.State.CORRUPTED,
        )
    }
    assert space.fidelity_name is not None
    assert space.fidelity is not None
    train_configs = encoder.encode([t.config for t in selected.values()], device=device)
    ids = torch.tensor(
        [int(config_id.split("_", maxsplit=1)[0]) for config_id in selected.keys()],
        device=device,
        dtype=torch.float64,
    )
    ids = ids + 1  # We add one to all ids to make room for the test configurations
    train_fidelities = torch.tensor(
        [t.config[space.fidelity_name] for t in selected.values()],
        device=device,
        dtype=torch.float64,
    )
    train_budgets = budget_domain.cast(train_fidelities, frm=space.fidelity.domain)
    X = _tokenize(
        ids=torch.tensor(ids, device=device), budgets=train_budgets, configs=train_configs
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
    i1 = torch.argsort(x[:, 1])
    i2 = i1[torch.argsort(x[i1][:, 0], stable=True)]
    sorted_x = x[i2]

    # Now that it's sorted, we essentially want to count the occurence of each id into counts
    _, counts = torch.unique_consecutive(sorted_x[:, 0], return_counts=True)

    # Now we can use these counts to get to the last occurence of each id
    # The -1 is because we want to index from 0 but sum starts at 1.
    ii = counts.cumsum(0) - 1
    return sorted_x[ii]


def _acquire_pfn(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    ftpfn: FTPFNSurrogate,
    y_to_beat: float,
    how: Literal["pi", "ei", "ucb", "lcb"],
) -> torch.Tensor:
    match how:
        case "pi":
            y_best = torch.full(
                size=(len(test_x),), fill_value=y_to_beat, dtype=FTPFN_DTYPE
            )
            return ftpfn.get_pi(train_x, train_y, test_x, y_best=y_best)
        case "ei":
            y_best = torch.full(
                size=(len(test_x),), fill_value=y_to_beat, dtype=FTPFN_DTYPE
            )
            return ftpfn.get_ei(train_x, train_y, test_x, y_best=y_best)
        case "ucb":
            y_best = torch.full(
                size=(len(test_x),), fill_value=y_to_beat, dtype=FTPFN_DTYPE
            )
            return ftpfn.get_ucb(train_x, train_y, test_x)
        case "lcb":
            return ftpfn.get_lcb(train_x, train_y, test_x)
        case _:
            raise ValueError(f"Unknown acquisition function {how}")


class IFBO(BaseOptimizer):
    """Base class for MF-BO algorithms that use DyHPO-like acquisition and budgeting."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        step_size: int | float = 1,
        use_priors: bool = False,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
        # arguments for model
        surrogate_model_args: dict | None = None,
        initial_design_size: int | None = None,
        n_acquisition_new_configs: int = 1_000,
        device: torch.device | None = None,
        **kwargs: Any,  # TODO: Remove this
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
        self.n_initial_design = initial_design_size
        self.n_acquisition_new_configs = n_acquisition_new_configs
        self.surrogate_model_args = surrogate_model_args or {}

        self._min_budget: int | float = space.fidelity.lower
        self._max_budget: int | float = space.fidelity.upper
        self._fidelity_name: str = space.fidelity_name
        self._initial_design: list[dict[str, Any]] | None = None

        params = {**space.numerical, **space.categoricals}
        self._prior = Prior.from_parameters(params) if use_priors else None
        self._ftpfn_encoder: TensorEncoder = TensorEncoder.default(
            params,
            # FTPFN doesn't support categoricals and we were recomenned to just evenly distribute
            # in the unit norm
            custom_transformers={
                cat_name: CategoricalToUnitNorm(choices=cat.choices)
                for cat_name, cat in space.categoricals.items()
            },
        )

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
        seed: int | None = None,
    ) -> SampledConfig:
        if seed is not None:
            raise NotImplementedError("Seed is not yet implemented for IFBO")

        ids = [int(config_id.split("_", maxsplit=1)[0]) for config_id in trials.keys()]
        new_id = max(ids) + 1 if len(ids) > 0 else 0

        # If we havn't passed the intial design phase
        if self._initial_design is None:
            self._initial_design = make_initial_design(
                space=self.pipeline_space,
                encoder=self._ftpfn_encoder,
                sample_default_first=self.sample_default_first,
                sampler="sobol" if self._prior is None else self._prior,
                seed=seed,
                sample_fidelity="min",
                sample_size=(
                    "ndim" if self.n_initial_design is None else self.n_initial_design
                ),
            )

        if new_id < len(self._initial_design):
            return SampledConfig(id=f"{new_id}_0", config=self._initial_design[new_id])

        # Otherwise, we proceed to surrogate phase
        ftpfn = FTPFNSurrogate(
            target_path=self.surrogate_model_args.get("target_path", None),
            version=self.surrogate_model_args.get("version", "0.0.1"),
            device=self.device,
        )
        x_train, maximize_ys = _encode_for_ftpfn(
            trials=trials,
            encoder=self._ftpfn_encoder,
            space=self.pipeline_space,
            budget_domain=self._budget_domain,
            device=self.device,
        )
        # PFN uses `0` id for test configurations, we remove this later
        x_train[:, 1] = x_train[:, 1] + 1

        # Fantasize the result of pending trials
        is_pending = maximize_ys.isnan()
        maximize_ys[is_pending] = ftpfn.get_mean_performance(
            train_x=x_train[~is_pending],
            train_y=maximize_ys[~is_pending],
            test_x=x_train[is_pending],
        )

        # We then sample a horizon, minimum one budget index increment and cast
        # to the budget domain expected by the ftpfn model
        rng = np.random.RandomState(seed)
        lower_index = self._budget_ix_domain.lower
        upper_index = self._budget_ix_domain.upper
        horizon = self._budget_domain.cast_one(
            rng.randint(lower_index, upper_index) + 1,
            frm=self._budget_ix_domain,
        )

        # Now we sample some new configurations into the domain expected by the FTPFN
        if self._prior is not None:
            acq_sampler = self._prior
        else:
            acq_sampler = Sampler.uniform(ndim=self._ftpfn_encoder.ncols)

        new_acq_configs = acq_sampler.sample(
            self.n_acquisition_new_configs,
            to=self._ftpfn_encoder.domains,
            device=self.device,
            seed=None,  # TODO
        )
        acq_new = _tokenize(
            ids=torch.zeros(self.n_acquisition_new_configs, device=self.device),
            budgets=torch.full(
                size=(self.n_acquisition_new_configs,),
                fill_value=self._budget_domain.lower,
                device=self.device,
            ),
            configs=new_acq_configs,
        )

        # Construct all our samples for acqusition:
        # 1. Take all non-pending configs
        acq_continue_existing = x_train[~is_pending].clone().detach()

        # 2. We only want to include the configuration at their highest
        # budget evaluated, i.e. don't include config_0_0 if config_0_1 is highest
        acq_continue_existing = _keep_highest_budget_evaluation(acq_continue_existing)

        # 3. Sub select all that are not fully evaluated
        acq_continue_existing = acq_continue_existing[acq_continue_existing[:, 1] < 1]

        # 4. Add in the new sampled configurations
        acq_samples = torch.vstack([acq_continue_existing, acq_new])

        # 5. Add on the horizon to the budget
        unclamped_budgets = acq_samples[:, 1] + horizon

        # 6. Clamp to the maximum of the budget domain
        acq_samples[:, 1] = torch.clamp(unclamped_budgets, max=self._budget_domain.upper)

        # Now get the PI of these samples according to MFPI_Random
        maximize_best_y = maximize_ys.max().item()
        lu = 10 ** rng.uniform(-4, -1)
        f_inc = maximize_best_y * (1 - lu)

        acq_scores = _acquire_pfn(
            train_x=x_train,
            train_y=maximize_ys[~is_pending],
            test_x=acq_samples,
            ftpfn=ftpfn,
            y_to_beat=f_inc,
            how="pi",
        )

        # Extract out the row which had the best PI
        best_ix = acq_scores.argmax()
        best_id = int(acq_samples[best_ix, 0].round().item())
        best_vector = acq_samples[best_ix, 2:].unsqueeze(0)
        best_config = self._ftpfn_encoder.unpack(best_vector)[0]

        if best_id == 0:
            # A newly sampled configuration was deemed more promising
            config_id = f"{new_id}_0"
            best_config[self._fidelity_name] = self._min_budget
            previous_config_id = None
            return SampledConfig(config_id, best_config, previous_config_id)

        # To get to the next fidelity value to provide,
        # 1. Get the budget before we added the horizon
        budget = float(unclamped_budgets[best_ix] - horizon)

        # 2. Cast to budget index domain
        budget_ix = self._budget_ix_domain.cast_one(budget, frm=self._budget_domain)

        # 3. Increment it to the next budget index
        budget_ix += 1

        # 4. And finally convert back into the fidelity domain
        fid_value = self._fid_domain.cast_one(budget_ix, frm=self._budget_ix_domain)

        real_best_id = best_id - 1  # NOTE: Remove the +1 we added to all ids earlier
        best_config[self._fidelity_name] = fid_value

        config_id = f"{real_best_id}_{budget_ix}"
        previous_config_id = f"{real_best_id}_{budget_ix - 1}"
        return SampledConfig(config_id, best_config, previous_config_id)
