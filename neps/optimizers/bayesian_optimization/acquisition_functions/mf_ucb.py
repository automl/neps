from typing import Any, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ....optimizers.utils import map_real_hyperparameters_from_tabular_ids
from ....search_spaces.search_space import IntegerParameter, SearchSpace
from ...multi_fidelity.utils import MFObservedData
from .mf_ei import MFStepBase
from .ucb import UpperConfidenceBound


# NOTE: the order of inheritance is important
class MF_UCB(MFStepBase, UpperConfidenceBound):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        surrogate_model_name: str = None,
        beta: float = 1.0,
        maximize: bool = False,
    ):
        """Upper Confidence Bound (UCB) acquisition function.

        Args:
            beta: Controls the balance between exploration and exploitation.
            maximize: If True, maximize the given model, else minimize.
                DEFAULT=False, assumes minimzation.
        """
        super().__init__(beta, maximize)
        self.pipeline_space = pipeline_space
        self.surrogate_model_name = surrogate_model_name
        self.surrogate_model = None
        self.observations = None
        self.b_step = None

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
        betas = []
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
                    # CAN ADAPT BETA PER-SAMPLE HERE
                    betas.append(self.beta)
                else:
                    # if the target_fidelity higher than the max drop the configuration
                    indices_to_drop.append(i)
            else:
                config.fidelity.value = target_fidelity
                budget_list.append(self.get_budget_level(config))
                # CAN ADAPT BETA PER-SAMPLE HERE
                betas.append(self.beta)

        # Drop unused configs
        x.drop(labels=indices_to_drop, inplace=True)

        return x, torch.Tensor(betas)

    def preprocess_gp(
        self, x: pd.Series, surrogate_name: str = "gp"
    ) -> Tuple[pd.Series, torch.Tensor]:
        if surrogate_name == "gp":
            x, inc_list = self.preprocess(x)
            return x, inc_list
        elif surrogate_name in ["deep_gp", "dpl"]:
            x, inc_list = self.preprocess(x)
            x_lcs = []
            for idx in x.index:
                if idx in self.observations.df.index.levels[0]:
                    # extracting the available/observed learning curve
                    lc = self.observations.extract_learning_curve(idx, budget_id=None)
                else:
                    # initialize a learning curve with a placeholder
                    # This is later padded accordingly for the Conv1D layer
                    lc = []
                x_lcs.append(lc)
            self.surrogate_model.set_prediction_learning_curves(x_lcs)
            return x, inc_list
        else:
            raise ValueError(f"Unrecognized surrogate model name: {surrogate_name}")

    def eval_pfn_ucb(
        self, x: Iterable, beta: float = (1 - 0.682) / 2
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """PFN-UCB modified to preprocess samples and accept list of incumbents."""
        ucb = self.surrogate_model.get_ucb(
            x_test=x.to(self.surrogate_model.device),
            beta=beta,  # TODO: extend to have different betas for each candidates in x
        )
        if len(ucb.shape) == 2:
            ucb = ucb.flatten()
        return ucb

    def eval(self, x: pd.Series, asscalar: bool = False) -> Tuple[np.ndarray, pd.Series]:
        if self.surrogate_model_name == "pfn":
            _x, _x_tok, _ = self.preprocess_pfn(x.copy())
            ucb = self.eval_pfn_ucb(_x_tok)
        elif self.surrogate_model_name in ["deep_gp", "gp", "dpl"]:
            _x, betas = self.preprocess_gp(x.copy(), self.surrogate_model_name)
            ucb = super().eval(_x.values.tolist(), betas, asscalar)
        else:
            raise ValueError(
                f"Unrecognized surrogate model name: {self.surrogate_model_name}"
            )

        return ucb, _x


class MF_UCB_AtMax(MF_UCB):
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
        betas = []
        for i, config in x.items():
            target_fidelity = config.fidelity.upper  # change from MFEI

            if config.fidelity.value == target_fidelity:
                # if the target_fidelity already reached, drop the configuration
                indices_to_drop.append(i)
            else:
                config.fidelity.value = target_fidelity
                budget_list.append(self.get_budget_level(config))

                # CAN ADAPT BETA PER-SAMPLE HERE
                betas.append(self.beta)

        # drop unused configs
        x.drop(labels=indices_to_drop, inplace=True)

        return x, torch.Tensor(betas)


class MF_UCB_Dyna(MF_UCB):
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
        pseudo_z_max = (
            self.b_step * pseudo_z_level_max + self.pipeline_space.fidelity.lower
        )

        # TODO: compare with this first draft logic
        # def update_fidelity(config):
        #     ### DO NOT DELETE THIS FUNCTION YET
        #     # for all configs, set the min(max(current fidelity + step, z_inc), pseudo_z_max)
        #     ## that is, choose the next highest marker from 1 and 2
        #     z_extrapolate = min(
        #         max(config.fidelity.value + self.b_step, z_inc),
        #         pseudo_z_max
        #     )
        #     config.fidelity.value = z_extrapolate
        #     return config

        def update_fidelity(config):
            # for all configs, set to pseudo_z_max
            ## that is, choose the highest seen fidelity in observation history
            z_extrapolate = pseudo_z_max
            config.fidelity.value = z_extrapolate
            return config

        # collect IDs for partial configurations
        _partial_config_ids = x.index <= max(self.observations.seen_config_ids)
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

        # CAN ADAPT BETA PER-SAMPLE HERE
        betas = [self.beta] * len(x)  # TODO: have tighter order check to Pd.Series index

        return x, torch.Tensor(betas)
