# type: ignore
from pathlib import Path
from typing import Any, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal

from copy import deepcopy

from neps.optimizers.utils import map_real_hyperparameters_from_tabular_ids
from neps.search_spaces.search_space import SearchSpace
from neps.optimizers.multi_fidelity.utils import MFObservedData
from neps.optimizers.bayesian_optimization.acquisition_functions.ei import ComprehensiveExpectedImprovement
from neps.optimizers.bayesian_optimization.acquisition_functions.mf_ei import MFStepBase


# NOTE: the order of inheritance is important
class MFPI(MFStepBase, ComprehensiveExpectedImprovement):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        surrogate_model_name: str = None,
        augmented_ei: bool = False,
        xi: float = 0.0,
        in_fill: str = "best",
        log_ei: bool = False,
    ):
        super().__init__(augmented_ei, xi, in_fill, log_ei)
        self.pipeline_space = pipeline_space
        self.surrogate_model_name = surrogate_model_name
        self.surrogate_model = None
        self.observations = None
        self.b_step = None

    def preprocess_inc_list(self, **kwargs) -> list:
        assert "budget_list" in kwargs, "Requires a list of query step for candidate set."
        budget_list = kwargs["budget_list"]
        performances = self.observations.get_best_performance_for_each_budget()
        inc_list = []
        for budget_level in budget_list:
            if budget_level in performances.index:
                inc = performances[budget_level]
            else:
                inc = self.observations.get_best_seen_performance()
            inc_list.append(inc)
        return inc_list

    def preprocess(self, x: pd.Series) -> Tuple[pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity Expected Improvement acquisition function.
        """
        budget_list = []
        if self.pipeline_space.has_tabular:
            # preprocess tabular space differently
            # expected input: IDs pertaining to the tabular data
            x = map_real_hyperparameters_from_tabular_ids(x, self.pipeline_space)
        indices_to_drop = []
        for i, config in x.items():
            target_fidelity = config.fidelity.lower
            if i <= max(self.observations.seen_config_ids):
                # IMPORTANT to set the fidelity at which EI will be calculated only for
                # the partial configs that have been observed already
                target_fidelity = config.fidelity.value + self.b_step

                if np.less_equal(target_fidelity, config.fidelity.upper):
                    # only consider the configs with fidelity lower than the max fidelity
                    config.update_hp_values({config.fidelity_name: target_fidelity})
                    budget_list.append(self.get_budget_level(config))
                else:
                    # if the target_fidelity higher than the max drop the configuration
                    indices_to_drop.append(i)
            else:
                config.update_hp_values({config.fidelity_name: target_fidelity})
                budget_list.append(self.get_budget_level(config))

        # Drop unused configs
        x.drop(labels=indices_to_drop, inplace=True)

        # Collecting incumbent list per configuration
        inc_list = self.preprocess_inc_list(budget_list=budget_list)

        return x, torch.Tensor(inc_list)

    def eval(self, x: pd.Series, asscalar: bool = False) -> Tuple[np.ndarray, pd.Series]:
        # deepcopy
        _x = pd.Series([deepcopy(x.loc[idx]) for idx in x.index.values], index=x.index)
        if self.surrogate_model_name == "ftpfn":
            _x, _x_tok, inc_list = self.preprocess_pfn(
                _x
            )  # IMPORTANT change from vanilla-EI
            pi = self.eval_pfn_pi(_x_tok, inc_list)
        else:
            raise ValueError(
                f"Unrecognized surrogate model name: {self.surrogate_model_name}"
            )

        if pi.is_cuda:
            pi = pi.cpu()
        if len(_x) > 1 and asscalar:
            return pi.detach().numpy(), _x
        else:
            return pi.detach().numpy().item(), _x

    def eval_pfn_pi(
        self, x: Iterable, inc_list: Iterable
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """PFN-PI modified to preprocess samples and accept list of incumbents."""
        pi = self.surrogate_model.get_pi(x.to(self.surrogate_model.device), inc_list)
        if len(pi.shape) == 2:
            pi = pi.flatten()
        return pi


class MFPI_AtMax(MFPI):

    def preprocess_inc_list(self, **kwargs) -> list:
        assert "len_x" in kwargs, "Requires the length of the candidate set."
        len_x = kwargs["len_x"]
        # finds global incumbent
        inc_value = min(self.observations.get_best_performance_for_each_budget())
        # uses the best seen value as the incumbent in EI computation for all candidates
        inc_list = [inc_value] * len_x
        return inc_list

    def preprocess(self, x: pd.Series) -> Tuple[pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point.
        Unlike the base class MFPI, sets the target fidelity to be max budget and the
        incumbent choice to be the max seen across history for all candidates.
        """
        budget_list = []
        if self.pipeline_space.has_tabular:
            # preprocess tabular space differently
            # expected input: IDs pertaining to the tabular data
            x = map_real_hyperparameters_from_tabular_ids(x, self.pipeline_space)

        indices_to_drop = []
        for i, config in x.items():
            target_fidelity = config.fidelity.upper  # change from MFEI

            if config.fidelity.value == target_fidelity:
                # if the target_fidelity already reached, drop the configuration
                indices_to_drop.append(i)
            else:
                config.update_hp_values({config.fidelity_name: target_fidelity})
                budget_list.append(self.get_budget_level(config))

        # drop unused configs
        x.drop(labels=indices_to_drop, inplace=True)

        # create the same incumbent for all candidates
        inc_list = self.preprocess_inc_list(len_x=len(x.index.values))

        return x, torch.Tensor(inc_list)


class MFPI_Dyna(MFPI_AtMax):
    """
    Computes extrapolation length of curves to maximum fidelity seen.
    Uses the global incumbent as the best score in EI computation.
    """

    def preprocess(self, x: pd.Series) -> Tuple[pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point.
        Unlike the base class MFEI, sets the target fidelity to be max budget and the
        incumbent choice to be the max seen across history for all candidates.
        """
        if self.pipeline_space.has_tabular:
            # preprocess tabular space differently
            # expected input: IDs pertaining to the tabular data
            x = map_real_hyperparameters_from_tabular_ids(x, self.pipeline_space)

        # find the maximum observed steps per config to obtain the current pseudo_z_max
        max_z_level_per_x = self.observations.get_max_observed_fidelity_level_per_config()
        pseudo_z_level_max = max_z_level_per_x.max()  # highest seen fidelity step so far
        # find the fidelity step at which the best seen performance was recorded
        z_inc_level = self.observations.get_budget_level_for_best_performance()
        # retrieving actual fidelity values from budget level
        ## marker 1: the fidelity value at which the best seen performance was recorded
        z_inc = self.b_step * z_inc_level + self.pipeline_space.fidelity.lower
        ## marker 2: the maximum fidelity value recorded in observation history
        pseudo_z_max = self.b_step * pseudo_z_level_max + self.pipeline_space.fidelity.lower

        def update_fidelity(config):
            # for all configs, set to pseudo_z_max
            ## that is, choose the highest seen fidelity in observation history
            z_extrapolate = pseudo_z_max
            config.update_hp_values({config.fidelity_name: z_extrapolate})
            return config

        # collect IDs for partial configurations
        _partial_config_ids = (x.index <= max(self.observations.seen_config_ids))
        # filter for configurations that reached max budget
        indices_to_drop = [
            _idx
            for _idx, _x in x.loc[_partial_config_ids].items()
            if _x.fidelity.value == self.pipeline_space.fidelity.upper
        ]
        # drop unused configs
        x.drop(labels=indices_to_drop, inplace=True)

        # set fidelity for all partial configs
        x = x.apply(update_fidelity)

        # create the same incumbent for all candidates
        inc_list = self.preprocess_inc_list(len_x=len(x.index.values))

        return x, torch.Tensor(inc_list)


class MFPI_Random(MFPI):

    BUDGET = 1000

    def __init__(
        self,
        pipeline_space: SearchSpace,
        horizon: str = "random",
        threshold: str = "random",
        surrogate_model_name: str = None,
        augmented_ei: bool = False,
        xi: float = 0.0,
        in_fill: str = "best",
        log_ei: bool = False,
    ):
        super().__init__(pipeline_space, surrogate_model_name, augmented_ei, xi, in_fill, log_ei)
        self.horizon = horizon
        self.threshold = threshold



    def set_state(
        self,
        pipeline_space: SearchSpace,
        surrogate_model: Any,
        observations: MFObservedData,
        b_step: Union[int, float],
        **kwargs,
    ):
        # set RNG
        self.rng = np.random.RandomState(seed=42)
        for i in range(len(observations.completed_runs)):
            self.rng.uniform(-4,-1)
            self.rng.randint(1,51)

        return super().set_state(pipeline_space, surrogate_model, observations, b_step)

    def sample_horizon(self, steps_passed):
        if self.horizon == 'random':
            shortest = self.pipeline_space.fidelity.lower
            longest = min(self.pipeline_space.fidelity.upper, self.BUDGET - steps_passed)
            return self.rng.randint(shortest, longest+1)
        elif self.horizon == 'max':
            return min(self.pipeline_space.fidelity.upper, self.BUDGET - steps_passed)
        else:
            return int(self.horizon)

    def sample_threshold(self, f_inc):
        if self.threshold == 'random':
            lu = 10**self.rng.uniform(-4,-1) # % of gap closed
        else:
            lu = float(self.threshold)
        return f_inc * (1 - lu)

    def preprocess(self, x: pd.Series) -> Tuple[pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity Expected Improvement acquisition function.
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
        t_value = self.sample_threshold(inc_value)
        inc_value = t_value

        # Like MFEI: set fidelities to query using horizon as self.b_step
        # Extension: Unlike DyHPO, we sample the horizon randomly over the full range
        horizon = self.sample_horizon(steps_passed)
        for i, config in x.items():
            if i <= max(self.observations.seen_config_ids):
                current_fidelity = config.fidelity.value
                if np.equal(config.fidelity.value, config.fidelity.upper):
                    # this training run has ended, drop it from future selection
                    indices_to_drop.append(i)
                else:
                    # a candidate partial training run to continue
                    target_fidelity = config.fidelity.value + horizon
                    config.update_hp_values({
                        config.fidelity_name: min(
                            config.fidelity.value + horizon, config.fidelity.upper
                        )  # if horizon exceeds max, query at max
                    }) 
                    inc_list.append(inc_value)
            else:
                # a candidate new training run that we would need to start
                current_fidelity = 0
                config.update_hp_values({config.fidelity_name: horizon})
                inc_list.append(inc_value)

        # Drop unused configs
        x.drop(labels=indices_to_drop, inplace=True)

        assert len(inc_list) == len(x)

        return x, torch.Tensor(inc_list)


class MFPI_Random_HiT(MFPI):

    BUDGET = 1000  # total budget in freeze-thaw steps available

    def set_state(
        self,
        pipeline_space: SearchSpace,
        surrogate_model: Any,
        observations: MFObservedData,
        b_step: Union[int, float],
        **kwargs,
    ):
        # set RNG
        self.rng = np.random.RandomState(seed=42)
        for i in range(len(observations.completed_runs)):
            self.rng.uniform(-4,0)
            self.rng.randint(1,51)

        return super().set_state(pipeline_space, surrogate_model, observations, b_step)

    def sample_horizon(self, steps_passed):
        shortest = self.pipeline_space.fidelity.lower
        longest = min(self.pipeline_space.fidelity.upper, self.BUDGET - steps_passed)
        return self.rng.randint(shortest, longest+1)

    def sample_threshold(self, f_inc):
        lu = 10**self.rng.uniform(-4,0) # % of gap closed
        return f_inc * (1 - lu)

    def preprocess(self, x: pd.Series) -> Tuple[pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity Expected Improvement acquisition function.
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
        t_value = self.sample_threshold(inc_value)
        inc_value = t_value

        # Like MFEI: set fidelities to query using horizon as self.b_step
        # Extension: Unlike DyHPO, we sample the horizon randomly over the full range
        horizon = self.sample_horizon(steps_passed)
        for i, config in x.items():
            if i <= max(self.observations.seen_config_ids):
                current_fidelity = config.fidelity.value
                if np.equal(config.fidelity.value, config.fidelity.upper):
                    # this training run has ended, drop it from future selection
                    indices_to_drop.append(i)
                else:
                    # a candidate partial training run to continue
                    target_fidelity = config.fidelity.value + horizon
                    # if horizon exceeds max, query at max
                    config.update_hp_values({config.fidelity_name: min(
                        config.fidelity.value + horizon, config.fidelity.upper
                    )})
                    inc_list.append(inc_value)
            else:
                # a candidate new training run that we would need to start
                current_fidelity = 0
                config.update_hp_values({config.fidelity_name: horizon})
                inc_list.append(inc_value)

        # Drop unused configs
        x.drop(labels=indices_to_drop, inplace=True)

        assert len(inc_list) == len(x)

        return x, torch.Tensor(inc_list)
