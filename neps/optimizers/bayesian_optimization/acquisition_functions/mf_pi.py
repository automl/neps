# type: ignore
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from neps.optimizers.bayesian_optimization.acquisition_functions.base_acquisition import (
    BaseAcquisition,
)
from neps.optimizers.multi_fidelity.utils import (
    MFObservedData,
    get_freeze_thaw_normalized_step,
    get_tokenized_data,
)
from neps.optimizers.utils import map_real_hyperparameters_from_tabular_ids

if TYPE_CHECKING:
    import pandas as pd

    from neps.search_spaces.search_space import SearchSpace


class MFPI(BaseAcquisition):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        surrogate_model_name: str | None = None,
    ):
        super().__init__()
        self.pipeline_space = pipeline_space
        self.surrogate_model_name = surrogate_model_name
        self.surrogate_model = None
        self.observations = None
        self.b_step = None

    def set_state(
        self,
        pipeline_space: SearchSpace,
        surrogate_model: Any,
        observations: MFObservedData,
        b_step: int | float,
        **kwargs,
    ):
        # overload to select incumbent differently through observations
        self.pipeline_space = pipeline_space
        self.surrogate_model = surrogate_model
        self.observations = observations
        self.b_step = b_step

    def preprocess(self, x: pd.Series) -> tuple[pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity Expected Improvement acquisition function.
        """
        raise NotImplementedError

    def eval(self, x: pd.Series, asscalar: bool = False) -> tuple[np.ndarray, pd.Series]:
        # deepcopy
        # _x = pd.Series([deepcopy(x.loc[idx]) for idx in x.index.values], index=x.index)
        if self.surrogate_model_name == "ftpfn":
            # preprocesses configs to have the appropriate fidelity values for acquisition
            _x, inc_list = self.preprocess(x.copy())
            _x_tok = get_tokenized_data(_x.values, ignore_fidelity=True)
            # padding IDs
            _idx = torch.Tensor(_x.index.values + 1)
            idx_mask = np.where(_idx > max(self.observations.seen_config_ids))[0]
            _idx[idx_mask] = 0
            # normalizing steps
            _steps = torch.Tensor(
                [
                    get_freeze_thaw_normalized_step(
                        _conf.fidelity.value,
                        self.pipeline_space.fidelity.lower,
                        self.pipeline_space.fidelity.upper,
                        self.b_step,
                    )
                    for _conf in _x
                ]
            )
            _x_tok = torch.hstack(
                ((_idx).reshape(-1, 1), _steps.reshape(-1, 1), torch.Tensor(_x_tok))
            )
            pi = self.eval_pfn_pi(_x_tok, inc_list)
        else:
            raise ValueError(
                f"Unrecognized surrogate model name: {self.surrogate_model_name}"
            )
        if pi.is_cuda:
            pi = pi.cpu()
        if len(_x) > 1 and asscalar:
            return pi.detach().numpy(), _x
        return pi.detach().numpy().item(), _x

    def eval_pfn_pi(
        self, x: Iterable, inc_list: Iterable
    ) -> np.ndarray | torch.Tensor | float:
        """PFN-PI modified to preprocess samples and accept list of incumbents."""
        pi = self.surrogate_model.get_pi(x.to(self.surrogate_model.device), inc_list)
        if len(pi.shape) == 2:
            pi = pi.flatten()
        return pi


class MFPI_Random(MFPI):
    BUDGET = 1000

    def __init__(
        self,
        pipeline_space: SearchSpace,
        horizon: str = "random",
        threshold: str = "random",
        surrogate_model_name: str | None = None,
    ):
        super().__init__(pipeline_space, surrogate_model_name)
        self.horizon = horizon
        self.threshold = threshold

    def set_state(
        self,
        pipeline_space: SearchSpace,
        surrogate_model: Any,
        observations: MFObservedData,
        b_step: int | float,
        seed: int = 42,
    ) -> None:
        # set RNG
        self.rng = np.random.RandomState(seed=seed)

        # TODO: wut is this?
        for _i in range(len(observations.completed_runs)):
            self.rng.uniform(-4, -1)
            self.rng.randint(1, 51)

        return super().set_state(pipeline_space, surrogate_model, observations, b_step)

    def sample_horizon(self, steps_passed):
        if self.horizon == "random":
            shortest = self.pipeline_space.fidelity.lower
            longest = min(self.pipeline_space.fidelity.upper, self.BUDGET - steps_passed)
            return self.rng.randint(shortest, longest + 1)
        if self.horizon == "max":
            return min(self.pipeline_space.fidelity.upper, self.BUDGET - steps_passed)
        return int(self.horizon)

    def sample_performance_threshold(self, f_inc):
        if self.threshold == "random":
            lu = 10 ** self.rng.uniform(-4, -1)  # % of gap closed
        else:
            lu = float(self.threshold)
        return f_inc * (1 - lu)

    def preprocess(self, x: pd.Series) -> tuple[pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity acquisition function.
        """
        if self.pipeline_space.has_tabular:
            # preprocess tabular space differently
            # expected input: IDs pertaining to the tabular data
            x = map_real_hyperparameters_from_tabular_ids(x, self.pipeline_space)

        indices_to_drop = []
        inc_list = []

        steps_passed = len(self.observations.completed_runs)

        # Like EI-AtMax, use the global incumbent as a basis for the EI threshold
        inc_value = min(self.observations.get_best_performance_for_each_budget())

        # Extension: Add a random min improvement threshold to encourage high risk high gain
        t_value = self.sample_performance_threshold(inc_value)
        inc_value = t_value

        # Like MFEI: set fidelities to query using horizon as self.b_step
        # Extension: Unlike DyHPO, we sample the horizon randomly over the full range
        horizon = self.sample_horizon(steps_passed)

        for i, config in x.items():
            if i <= max(self.observations.seen_config_ids):
                if np.equal(config.fidelity.value, config.fidelity.upper):
                    # this training run has ended, drop it from future selection
                    indices_to_drop.append(i)
                else:
                    # a candidate partial training run to continue
                    config.update_hp_values(
                        {
                            config.fidelity_name: min(
                                config.fidelity.value + horizon, config.fidelity.upper
                            )  # if horizon exceeds max, query at max
                        }
                    )
                    inc_list.append(inc_value)
            else:
                # a candidate new training run that we would need to start
                config.update_hp_values({config.fidelity_name: horizon})
                inc_list.append(inc_value)

        # Drop unused configs
        x = x.drop(labels=indices_to_drop)

        assert len(inc_list) == len(x)

        return x, torch.Tensor(inc_list)
