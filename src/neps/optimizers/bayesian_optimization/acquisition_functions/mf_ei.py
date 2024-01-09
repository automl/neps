# type: ignore
from pathlib import Path
from typing import Any, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal

from copy import deepcopy

from ....optimizers.utils import map_real_hyperparameters_from_tabular_ids
from ....search_spaces.search_space import IntegerParameter, SearchSpace
from ...multi_fidelity.utils import MFObservedData
from .base_acquisition import BaseAcquisition
from .ei import ComprehensiveExpectedImprovement
from ....utils.common import SimpleCSVWriter


class MFStepBase(BaseAcquisition):
    """A class holding common operations that can be inherited.

    WARNING: Unsafe use of self attributes, can break if not used correctly.
    """
    def set_state(
        self,
        pipeline_space: SearchSpace,
        surrogate_model: Any,
        observations: MFObservedData,
        b_step: Union[int, float],
        **kwargs,
    ):
        # overload to select incumbent differently through observations
        self.pipeline_space = pipeline_space
        self.surrogate_model = surrogate_model
        self.observations = observations
        self.b_step = b_step
        return

    def get_budget_level(self, config) -> int:
        return int((config.fidelity.value - config.fidelity.lower) / self.b_step)


    def preprocess_gp(self, x: pd.Series) -> Tuple[pd.Series, torch.Tensor]:
        x, inc_list = self.preprocess(x)
        return x, inc_list

    def preprocess_deep_gp(self, x: pd.Series) -> Tuple[pd.Series, torch.Tensor]:
        x, inc_list = self.preprocess(x)
        x_lcs = []
        for idx in x.index:
            if idx in self.observations.df.index.levels[0]:
                # TODO: Samir, check if `budget_id=None` is okay?
                # budget_level = self.get_budget_level(x[idx])
                # extracting the available/observed learning curve
                lc = self.observations.extract_learning_curve(idx, budget_id=None)
            else:
                # initialize a learning curve with a placeholder
                # This is later padded accordingly for the Conv1D layer
                lc = []
            x_lcs.append(lc)
        self.surrogate_model.set_prediction_learning_curves(x_lcs)
        return x, inc_list

    def preprocess_pfn(self, x: pd.Series) -> Tuple[torch.Tensor, pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity Expected Improvement acquisition function.
        """
        _x, inc_list = self.preprocess(x.copy())
        _x_tok = self.observations.tokenize(_x, as_tensor=True)
        len_partial = len(self.observations.seen_config_ids)
        z_min = x[0].fidelity.lower
        z_max = x[0].fidelity.upper
        # converting fidelity to the discrete budget level
        # STRICT ASSUMPTION: fidelity is the second dimension
        _x_tok[:len_partial, 1] = (
            _x_tok[:len_partial, 1] + self.b_step - z_min
        ) / self.b_step
        _x_tok[:, 1] = _x_tok[:, 1] / z_max
        return _x, _x_tok, inc_list


# NOTE: the order of inheritance is important
class MFEI(MFStepBase, ComprehensiveExpectedImprovement):
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
                    config.fidelity.value = target_fidelity
                    budget_list.append(self.get_budget_level(config))
                else:
                    # if the target_fidelity higher than the max drop the configuration
                    indices_to_drop.append(i)
            else:
                config.fidelity.value = target_fidelity
                budget_list.append(self.get_budget_level(config))

        # Drop unused configs
        x.drop(labels=indices_to_drop, inplace=True)

        # Collecting incumbent list per configuration
        inc_list = self.preprocess_inc_list(budget_list=budget_list)

        return x, torch.Tensor(inc_list)

    def eval(self, x: pd.Series, asscalar: bool = False) -> Tuple[np.ndarray, pd.Series]:
        # deepcopy
        _x = pd.Series([x.loc[idx].copy() for idx in x.index.values], index=x.index)
        if self.surrogate_model_name == "pfn":
            _x, _x_tok, inc_list = self.preprocess_pfn(
                x.copy()
            )  # IMPORTANT change from vanilla-EI
            ei = self.eval_pfn_ei(_x_tok, inc_list)
        elif self.surrogate_model_name in ["deep_gp", "dpl"]:
            _x, inc_list = self.preprocess_deep_gp(
                _x
            )  # IMPORTANT change from vanilla-EI
            ei = self.eval_gp_ei(_x.values.tolist(), inc_list)
        elif self.surrogate_model_name == "gp":
            _x, inc_list = self.preprocess_gp(
                _x
            )  # IMPORTANT change from vanilla-EI
            ei = self.eval_gp_ei(_x.values.tolist(), inc_list)
        else:
            raise ValueError(
                f"Unrecognized surrogate model name: {self.surrogate_model_name}"
            )

        if ei.is_cuda:
            ei = ei.cpu()
        if len(_x) > 1 and asscalar:
            return ei.detach().numpy(), _x
        else:
            return ei.detach().numpy().item(), _x

    def eval_pfn_ei(
        self, x: Iterable, inc_list: Iterable
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """PFN-EI modified to preprocess samples and accept list of incumbents."""
        ei = self.surrogate_model.get_ei(x.to(self.surrogate_model.device), inc_list)
        if len(ei.shape) == 2:
            ei = ei.flatten()
        return ei

    def eval_gp_ei(
        self, x: Iterable, inc_list: Iterable
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """Vanilla-EI modified to preprocess samples and accept list of incumbents."""
        _x = x.copy()
        try:
            mu, cov = self.surrogate_model.predict(_x)
        except ValueError as e:
            raise e
            # return -1.0  # in case of error. return ei of -1
        std = torch.sqrt(torch.diag(cov))

        mu_star = inc_list.to(mu.device)  # IMPORTANT change from vanilla-EI

        gauss = Normal(torch.zeros(1, device=mu.device), torch.ones(1, device=mu.device))
        # u = (mu - mu_star - self.xi) / std
        # ei = std * updf + (mu - mu_star - self.xi) * ucdf
        if self.log_ei:
            # we expect that f_min is in log-space
            f_min = mu_star - self.xi
            v = (f_min - mu) / std
            ei = torch.exp(f_min) * gauss.cdf(v) - torch.exp(
                0.5 * torch.diag(cov) + mu
            ) * gauss.cdf(v - std)
        else:
            u = (mu_star - mu - self.xi) / std
            ucdf = gauss.cdf(u)
            updf = torch.exp(gauss.log_prob(u))
            ei = std * updf + (mu_star - mu - self.xi) * ucdf
            # Clip ei if std == 0.0
            ei = torch.where(torch.isclose(std, torch.tensor(0.0)), 0, ei)
        if self.augmented_ei:
            sigma_n = self.surrogate_model.likelihood
            ei *= 1.0 - torch.sqrt(torch.tensor(sigma_n, device=mu.device)) / torch.sqrt(
                sigma_n + torch.diag(cov)
            )

        # Save data for writing
        self.mu_star = mu_star.detach().numpy().tolist()
        self.mu = mu.detach().numpy().tolist()
        self.std = std.detach().numpy().tolist()
        return ei


class MFEI_AtMax(MFEI):

    def preprocess_inc_list(self, **kwargs) -> list:
        assert "len_x" in kwargs, "Requires the length of the candidate set."
        len_x = kwargs["len_x"]
        inc_value = min(self.observations.get_best_performance_for_each_budget())
        inc_list = [inc_value] * len_x
        return inc_list

    def preprocess(self, x: pd.Series) -> Tuple[pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point.
        Unlike the base class MFEI, sets the target fidelity to be max budget and the
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
                config.fidelity.value = target_fidelity
                budget_list.append(self.get_budget_level(config))

        # drop unused configs
        x.drop(labels=indices_to_drop, inplace=True)

        # create the same incumbent for all candidates
        inc_list = self.preprocess_inc_list(len_x=len(x.index.values))

        return x, torch.Tensor(inc_list)


class MFEI_Dyna(MFEI_AtMax):

    def preprocess(self, x: pd.Series) -> Tuple[pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point.
        Unlike the base class MFEI, sets the target fidelity to be max budget and the
        incumbent choice to be the max seen across history for all candidates.
        """
        budget_list = []
        if self.pipeline_space.has_tabular:
            # preprocess tabular space differently
            # expected input: IDs pertaining to the tabular data
            x = map_real_hyperparameters_from_tabular_ids(x, self.pipeline_space)

        # find the maximum observed steps per config to obtain the running pseudo_z_max
        max_z_level_per_x = self.observations.get_max_observed_fidelity_level_per_config()
        pseudo_z_level_max = max_z_level_per_x.max()
        # find the fidelity step at which the best seen performance was recorded
        z_inc_level = self.observations.get_budget_level_for_best_performance()
        # retrieving actual fidelity values from budget level
        ## marker 1
        z_inc = self.b_step * z_inc_level + self.pipeline_space.fidelity.lower
        ## marker 2
        pseudo_z_max = self.b_step * pseudo_z_level_max + self.pipeline_space.fidelity.lower

        def update_fidelity(config):
            # for all configs, set the min(max(current fidelity + step, z_inc), pseudo_z_max)
            ## that is, choose the next highest marker from 1 and 2
            z_extrapolate = min(
                max(config.fidelity.value + self.b_step, z_inc),
                pseudo_z_max
            )
            config.fidelity.value = z_extrapolate
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
