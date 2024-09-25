from typing import Any, Mapping

import math
import numpy as np
import torch

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.bayesian_optimization.models.ftpfn import FTPFNSurrogate
from neps.optimizers.intial_design import make_initial_design
from neps.sampling.samplers import Sampler
from neps.search_spaces.domain import Domain
from neps.search_spaces.encoding import CategoricalToUnitNorm, TensorEncoder
from neps.search_spaces.search_space import FloatParameter, IntegerParameter, SearchSpace
from neps.state.trial import Trial
from neps.state.optimizer import BudgetInfo


# NOTE: Ifbo was trained using 32 bit
FTPFN_DTYPE = torch.float32


def tokenize(
    ids: torch.Tensor,
    budgets: torch.Tensor,
    configs: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([ids.unsqueeze(1), budgets.unsqueeze(1), configs], dim=1)


def _encode_ftpfn(
    trials: Mapping[str, Trial],
    encoder: TensorEncoder,
    space: SearchSpace,
    budget_domain: Domain,
    device: torch.device | None = None,
    dtype: torch.dtype = FTPFN_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    ids = ids
    train_fidelities = torch.tensor(
        [t.config[space.fidelity_name] for t in selected.values()],
        device=device,
        dtype=torch.float64,
    )
    train_budgets = budget_domain.cast(train_fidelities, frm=space.fidelity.domain)
    X = tokenize(
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


def _remove_duplicates(x: torch.Tensor) -> torch.Tensor:
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


class IFBO(BaseOptimizer):
    """Base class for MF-BO algorithms that use DyHPO-like acquisition and budgeting."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        step_size: int = 1,
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
            use_priors: Allows random samples to be generated from a default
                Samples generated from a Gaussian centered around the default value
            sampling_policy: The type of sampling procedure to use
            promotion_policy: The type of promotion procedure to use
            sample_default_first: Whether to sample the default configuration first
            initial_design_size: Number of configurations to sample before starting optimization

                If None, the number of configurations will be equal to the number of dimensions.

            device: Device to use for the model
        """
        assert pipeline_space.fidelity is not None
        assert isinstance(pipeline_space.fidelity_name, str)

        super().__init__(pipeline_space=pipeline_space)
        self.step_size = step_size
        self.use_priors = use_priors
        self.sample_default_first = sample_default_first
        self.sample_default_at_target = sample_default_at_target
        self.surrogate_model_args = surrogate_model_args or {}
        self.device = device
        self.n_initial_design: int | None = initial_design_size
        self.n_acquisition_new_configs = n_acquisition_new_configs

        self._min_budget: int | float = pipeline_space.fidelity.lower
        self._max_budget: int | float = pipeline_space.fidelity.upper
        self._fidelity_name: str = pipeline_space.fidelity_name
        self._ftpfn_encoder: TensorEncoder = TensorEncoder.default(
            {
                **self.pipeline_space.numerical,
                **self.pipeline_space.categoricals,
            },
            custom_transformers={
                cat_name: CategoricalToUnitNorm(choices=cat.choices)
                for cat_name, cat in self.pipeline_space.categoricals.items()
            },
        )
        self._initial_design: list[dict[str, Any]] | None = None

        # TODO: We want it to be evenly divided by step size, so we need
        # to add something to the minimum fidelity to ensure this.
        maybe_bins = math.ceil((self._max_budget - self._min_budget) / self.step_size) + 1
        match pipeline_space.fidelity:
            case IntegerParameter():
                assert pipeline_space.fidelity.domain.cardinality is not None
                bins = min(maybe_bins, pipeline_space.fidelity.domain.cardinality)
            case FloatParameter():
                bins = maybe_bins
            case _:
                raise NotImplementedError(
                    f"Fidelity type {type(pipeline_space.fidelity)} not supported"
                )

        # Domain of fidelity values, i.e. what is given in the configs that we
        # give to the user to evaluate at.
        self._fid_domain = pipeline_space.fidelity.domain

        # Domain in which we should pass budgets to ifbo model
        self._budget_domain = Domain.float(1 / self._max_budget, 1)

        # Domain from which we assign an index to each budget
        # Automatically takes care of rounding
        self._budget_index_domain = Domain.indices(bins)

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
                sampler="sobol",
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
        x_train, maximize_ys = _encode_ftpfn(
            trials=trials,
            encoder=self._ftpfn_encoder,
            space=self.pipeline_space,
            budget_domain=self._budget_domain,
            device=self.device,
        )
        x_train[:, 1] = x_train[:, 1] + 1  # PFN uses `0` id for test configurations

        # Get the best performance so far
        maximize_best_y = maximize_ys.max().item()

        # Fantasize the result of pending trials
        is_pending = maximize_ys.isnan()
        maximize_ys[is_pending] = ftpfn.get_mean_performance(
            train_x=x_train[~is_pending],
            train_y=maximize_ys[~is_pending],
            test_x=x_train[is_pending],
        )

        rng = np.random.RandomState(seed)
        uniform = Sampler.uniform(ndim=self._ftpfn_encoder.ncols)

        # We sample the horizon in terms of step numbers to take
        lower_index = self._budget_index_domain.lower
        upper_index = self._budget_index_domain.upper
        # The plus 1 here is because we want to sample that at least one step
        # should be taken.
        horizon_index_increment = rng.randint(lower_index, upper_index) + 1

        # We then normalize it to FTPFN normalized budget domain
        horizon = self._budget_domain.cast_one(
            horizon_index_increment,
            frm=self._budget_index_domain,
        )

        # We give them all the special 0 id, as well as set the budget accordinly
        acq_new = tokenize(
            ids=torch.zeros(self.n_acquisition_new_configs, device=self.device),
            budgets=torch.zeros(self.n_acquisition_new_configs, device=self.device),
            configs=uniform.sample(
                n=self.n_acquisition_new_configs,
                to=self._ftpfn_encoder.domains,
                seed=None,  # TODO
                device=self.device,
            ),
        )

        # Construct all our samples for acqusition:
        # 1. Take all non-pending configs
        acq_train = x_train[~is_pending].clone().detach()

        # 2. We only want to include the configuration rows
        #   that are at their highest budget,
        #   i.e. don't include config_0_0 and config_0_1
        acq_train = _remove_duplicates(acq_train)

        # 3. Sub select all that are at a partial budget i.e. can evaluate further
        #   Note, it's important to do this after the above
        partial_eval_mask = acq_train[:, 1] < 1
        acq_train = acq_train[partial_eval_mask]

        # 4. Add in the new sampled configurations
        acq_samples = torch.vstack([acq_train, acq_new])

        # 5. Add on the horizon to the budget, and clamping to maximum
        #     Note that we hold onto the intermediate unclamped budget for later
        unclamped_budgets = acq_samples[:, 1] + horizon
        acq_samples[:, 1] = torch.clamp(unclamped_budgets, max=1)

        # Now get the PI of these samples
        lu = 10 ** rng.uniform(-4, -1)
        f_inc = maximize_best_y * (1 - lu)
        n_acq_samples = len(acq_samples)
        pi_new_samples = ftpfn.get_pi(
            train_x=x_train,
            train_y=maximize_ys,
            test_x=acq_samples,
            y_best=torch.full(size=(n_acq_samples,), fill_value=f_inc, dtype=FTPFN_DTYPE),
        )
        best_ix = pi_new_samples.argmax()

        # Extract out the row which had the best PI
        best_id = int(acq_samples[best_ix, 0].round().item())
        best_vector = acq_samples[best_ix, 2:].unsqueeze(0)
        best_config = self._ftpfn_encoder.unpack(best_vector)[0]

        if best_id == 0:
            # A newly sampled configuration was deemed more promising
            config_id = f"{new_id}_0"
            best_config[self._fidelity_name] = self._min_budget
            previous_config_id = None
            return SampledConfig(config_id, best_config, previous_config_id)

        else:
            # To calculate the next step to take in fidelity space, we remove the horizon
            previous_budget_of_acquired_config = unclamped_budgets[best_ix] - horizon

            # Then we transform this:
            # 1. Back to budget_index space
            # 2. Increment it by one
            # 3. Transform back to fidelity space
            budget_ix = self._budget_index_domain.cast_one(
                float(previous_budget_of_acquired_config), frm=self._budget_domain
            )
            budget_ix += 1
            fid_value = self._fid_domain.cast_one(
                budget_ix, frm=self._budget_index_domain
            )

            real_best_id = best_id - 1  # NOTE: Remove the +1 we added to all ids
            best_config[self._fidelity_name] = fid_value

            config_id = f"{real_best_id}_{budget_ix}"
            previous_config_id = f"{real_best_id}_{budget_ix - 1}"

            return SampledConfig(config_id, best_config, previous_config_id)
