# type: ignore
from typing import Any, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal

from ....optimizers.utils import map_real_hyperparameters_from_tabular_ids
from ....search_spaces.search_space import SearchSpace
from ...multi_fidelity.utils import MFObservedData
from .ei import ComprehensiveExpectedImprovement


class MFEI(ComprehensiveExpectedImprovement):
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

    def get_budget_level(self, config) -> int:
        return int((config.fidelity.value - config.fidelity.lower) / self.b_step)

    def preprocess(self, x: pd.Series) -> Tuple[Iterable, Iterable]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity Expected Improvement acquisition function.
        """
        budget_list = []

        if self.pipeline_space.has_tabular:
            # preprocess tabular space differently
            # expected input: IDs pertaining to the tabular data
            # expected output: IDs pertaining to current observations and set of HPs
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
                    config.fidelity.set_value(target_fidelity)
                    budget_list.append(self.get_budget_level(config))
                else:
                    # if the target_fidelity higher than the max drop the configuration
                    indices_to_drop.append(i)
            else:
                config.fidelity.set_value(target_fidelity)
                budget_list.append(self.get_budget_level(config))

        # Drop unused configs
        x.drop(labels=indices_to_drop, inplace=True)

        performances = self.observations.get_best_performance_for_each_budget()
        inc_list = []
        for budget_level in budget_list:
            if budget_level in performances.index:
                inc = performances[budget_level]
            else:
                inc = self.observations.get_best_seen_performance()
            inc_list.append(inc)

        return x, torch.Tensor(inc_list)

    def preprocess_gp(self, x: Iterable) -> Tuple[Iterable, Iterable]:
        x, inc_list = self.preprocess(x)
        return x.values.tolist(), inc_list

    def preprocess_deep_gp(self, x: Iterable) -> Tuple[Iterable, Iterable]:
        x, inc_list = self.preprocess(x)
        x_lcs = []
        for idx in x.index:
            if idx in self.observations.df.index.levels[0]:
                budget_level = self.get_budget_level(x[idx])
                lc = self.observations.extract_learning_curve(idx, budget_level)
            else:
                # initialize a learning curve with a place holder
                # This is later padded accordingly for the Conv1D layer
                lc = [0.0]
            x_lcs.append(lc)
        self.surrogate_model.set_prediction_learning_curves(x_lcs)
        return x.values.tolist(), inc_list

    def preprocess_pfn(self, x: Iterable) -> Tuple[Iterable, Iterable, Iterable]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity Expected Improvement acquisition function.
        """
        _x, inc_list = self.preprocess(x.copy())
        _x_tok = self.observations.tokenize(_x, as_tensor=True)
        len_partial = len(self.observations.seen_config_ids)
        z_min = x[0].fidelity.lower
        # converting fidelity to the discrete budget level
        # STRICT ASSUMPTION: fidelity is the first dimension
        _x_tok[:len_partial, 0] = (
            _x_tok[:len_partial, 0] + self.b_step - z_min
        ) / self.b_step
        return _x_tok, _x, inc_list

    def eval(self, x: pd.Series, asscalar: bool = False) -> Tuple[np.ndarray, pd.Series]:
        # _x = x.copy()  # preprocessing needs to change the reference x Series so we don't copy here
        if self.surrogate_model_name == "pfn":
            _x_tok, _x, inc_list = self.preprocess_pfn(
                x.copy()
            )  # IMPORTANT change from vanilla-EI
            ei = self.eval_pfn_ei(_x_tok, inc_list)
        elif self.surrogate_model_name == "deep_gp":
            _x, inc_list = self.preprocess_deep_gp(
                x.copy()
            )  # IMPORTANT change from vanilla-EI
            ei = self.eval_gp_ei(_x, inc_list)
            _x = pd.Series(_x, index=np.arange(len(_x)))
        else:
            _x, inc_list = self.preprocess_gp(
                x.copy()
            )  # IMPORTANT change from vanilla-EI
            ei = self.eval_gp_ei(_x, inc_list)
            _x = pd.Series(_x, index=np.arange(len(_x)))

        if ei.is_cuda:
            ei = ei.cpu()
        if len(x) > 1 and asscalar:
            return ei.detach().numpy(), _x
        else:
            return ei.detach().numpy().item(), _x

    def eval_pfn_ei(
        self, x: Iterable, inc_list: Iterable
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """PFN-EI modified to preprocess samples and accept list of incumbents."""
        # x, inc_list = self.preprocess(x)  # IMPORTANT change from vanilla-EI
        # _x = x.copy()
        ei = self.surrogate_model.get_ei(x.to(self.surrogate_model.device), inc_list)
        if len(ei.shape) == 2:
            ei = ei.flatten()
        return ei

    def eval_gp_ei(
        self, x: Iterable, inc_list: Iterable
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """Vanilla-EI modified to preprocess samples and accept list of incumbents."""
        # x, inc_list = self.preprocess(x)  # IMPORTANT change from vanilla-EI
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
        if self.augmented_ei:
            sigma_n = self.surrogate_model.likelihood
            ei *= 1.0 - torch.sqrt(torch.tensor(sigma_n, device=mu.device)) / torch.sqrt(
                sigma_n + torch.diag(cov)
            )
        return ei

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
