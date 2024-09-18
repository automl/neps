from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.bayesian_optimization.acquisition_functions.mf_pi import MFPI_Random
from neps.optimizers.bayesian_optimization.acquisition_samplers.freeze_thaw_sampler import (
    FreezeThawSampler,
)
from neps.optimizers.multi_fidelity.mf_bo import PFNSurrogate
from neps.optimizers.multi_fidelity.utils import MFObservedData
from neps.search_spaces.search_space import FloatParameter, IntegerParameter, SearchSpace
from neps.state.trial import Trial

if TYPE_CHECKING:
    from neps.state.optimizer import BudgetInfo


def _adjust_fidelity_for_freeze_thaw_steps(
    pipeline_space: SearchSpace,
    step_size: int,
) -> SearchSpace:
    """Adjusts the fidelity range to be divisible by `step_size` for Freeze-Thaw."""
    assert pipeline_space.fidelity is not None

    # Check if the fidelity range is divided into equal sized steps by `step_size`
    fid_range = pipeline_space.fidelity.upper - pipeline_space.fidelity.lower
    remainder = fid_range % step_size
    if remainder == 0:
        return pipeline_space

    # Adjust the fidelity lower bound to be divisible by `step_size` into equal steps
    # Pushing the lower bound of the fidelity space by an offset to ensure equal-sized steps
    offset = step_size - remainder
    pipeline_space.fidelity.lower += offset

    warnings.warn(
        f"Adjusted fidelity lower bound to {pipeline_space.fidelity.lower} "
        f"for equal-sized steps of {step_size}.",
        UserWarning,
        stacklevel=3,
    )
    return pipeline_space


# TODO: Maybe make this a part of searchspace functionality
def get_budget_value(
    space: SearchSpace,
    step_size: int,
    budget_level: int | float,
) -> int | float:
    assert space.fidelity is not None
    match space.fidelity:
        case IntegerParameter():
            return int(step_size * budget_level + space.fidelity.lower)
        case FloatParameter():
            return step_size * budget_level + space.fidelity.lower
        case _:
            raise NotImplementedError(
                f"Fidelity parameter: {space.fidelity}"
                f"must be one of the types: "
                f"[IntegerParameter, FloatParameter], but is type:"
                f"{type(space.fidelity)}"
            )


class IFBO(BaseOptimizer):
    """Base class for MF-BO algorithms that use DyHPO-like acquisition and budgeting."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        step_size: int = 1,
        use_priors: bool = False,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
        patience: int = 100,
        # arguments for model
        surrogate_model_args: dict | None = None,
        initial_design_size: int = 1,
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
        """
        assert self.pipeline_space.fidelity is not None

        # Adjust pipeline space fidelity steps to be equally spaced
        pipeline_space = _adjust_fidelity_for_freeze_thaw_steps(pipeline_space, step_size)
        super().__init__(pipeline_space=pipeline_space, patience=patience)

        self.step_size = step_size
        self.use_priors = use_priors
        self.surrogate_model_args = surrogate_model_args
        self.sample_default_first = sample_default_first
        self.sample_default_at_target = sample_default_at_target

        self._initial_design_size = initial_design_size

        self.min_budget: int | float = self.pipeline_space.fidelity.lower
        self.max_budget: int | float = self.pipeline_space.fidelity.upper

        fidelity_name = self.pipeline_space.fidelity_name
        assert isinstance(fidelity_name, str)
        self.fidelity_name: str = fidelity_name

        self._model_update_failed = False

    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo,
        optimizer_state: dict[str, Any],
        seed: int | None = None,
    ) -> tuple[SampledConfig, dict[str, Any]]:
        if seed is not None:
            raise NotImplementedError("Seed is not yet implemented for IFBO")

        observed_configs = MFObservedData.from_trials(trials)

        in_initial_design_phase = (
            len(observed_configs.completed_runs) < self._initial_design_size
        )
        if in_initial_design_phase:
            # TODO: Copy BO setup where we can sample SOBOL or from Prior
            self.logger.debug("Sampling from initial design...")
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
            _config_dict = config.hp_values()
            _config_dict.update({self.fidelity_name: self.min_budget})
            config.set_hyperparameters_from_dict(_config_dict)
            _config_id = observed_configs.next_config_id()
            return SampledConfig(
                config=config.hp_values(), id=_config_id, previous_config_id=None
            ), optimizer_state

        # TODO: Maybe just remove `PFNSurrogate` as a whole and use FTPFN directly...
        #    this depends on whether we can actually create a proper surrogate model abstraction
        # TODO: Really all of this should just be passed into an __init__ instead of 3 stage process
        model_policy = PFNSurrogate(
            pipeline_space=self.pipeline_space,
            surrogate_model_args=self.surrogate_model_args,
            step_size=self.step_size,
        )
        model_policy.observed_configs = observed_configs
        model_policy.update_model()

        # TODO: Replace with more efficient samplers we have from BO
        # TODO: Just make this take in everything at __init__ instead of a 2 stage init
        acquisition_sampler = FreezeThawSampler(
            pipeline_space=self.pipeline_space, patience=self.patience
        )
        acquisition_sampler.set_state(
            self.pipeline_space, observed_configs, self.step_size
        )

        samples = acquisition_sampler.sample(set_new_sample_fidelity=self.min_budget)

        # TODO: See if we can get away from `set_state` style things
        # and just instantiate it with what it needs
        acquisition = MFPI_Random(
            pipeline_space=self.pipeline_space, surrogate_model_name="ftpfn"
        )
        acquisition.set_state(
            self.pipeline_space,
            model_policy.surrogate_model,
            observed_configs,
            self.step_size,
        )

        # `_samples` should have new configs with fidelities set to as required
        acq, _samples = acquisition.eval(x=samples, asscalar=True)
        # NOTE: len(samples) need not be equal to len(_samples) as `samples` contain
        # all (partials + new) configurations obtained from the sampler, but
        # in `_samples`, configs are removed that have reached maximum epochs allowed

        best_idx = acq.argmax()
        _config_id = best_idx

        # NOTE: `samples` and `_samples` should share the same index values, hence,
        # avoid using `.iloc` and work with `.loc` on these pandas DataFrame/Series
        config: SearchSpace = samples.loc[_config_id]
        config = config.clone()

        # IMPORTANT: setting the fidelity value appropriately
        if best_idx > max(observed_configs.seen_config_ids):
            next_fid_value = self.min_budget
        else:
            max_observed_fids = (
                observed_configs.get_max_observed_fidelity_level_per_config()
            )
            best_configs_max_fid = max_observed_fids.loc[best_idx]
            budget_value = get_budget_value(
                space=self.pipeline_space,
                step_size=self.step_size,
                budget_level=best_configs_max_fid,
            )
            next_fid_value = budget_value + self.step_size

        config.update_hp_values({self.fidelity_name: next_fid_value})

        # Lastly, we need to generate config id for it.
        budget_level = int(np.ceil((next_fid_value - self.min_budget) / self.step_size))
        if _config_id in observed_configs.seen_config_ids:
            config_id = f"{_config_id}_{budget_level}"
            previous_config_id = f"{_config_id}_{budget_value - 1}"
        else:
            config_id = f"{observed_configs.next_config_id()}_{budget_level}"

        return SampledConfig(
            config=config.hp_values(), id=config_id, previous_config_id=previous_config_id
        ), optimizer_state
