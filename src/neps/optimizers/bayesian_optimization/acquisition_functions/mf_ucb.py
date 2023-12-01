from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ....optimizers.utils import map_real_hyperparameters_from_tabular_ids
from ....search_spaces.search_space import SearchSpace
from ...multi_fidelity.utils import MFObservedData
from .ucb import UpperConfidenceBound


class MF_UCB(UpperConfidenceBound):
    def __init__(self,
        pipeline_space: SearchSpace,
        surrogate_model_name: str = None,
        beta: float=1.0,
        maximize: bool=False
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

    def get_budget_level(self, config) -> int:
        return int((config.fidelity.value - config.fidelity.lower) / self.b_step)

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
        elif surrogate_name == "deep_gp":
            x, inc_list = self.preprocess(x)
            x_lcs = []
            for idx in x.index:
                if idx in self.observations.df.index.levels[0]:
                    # extracting the available/observed learning curve
                    lc = self.observations.extract_learning_curve(idx, budget_id=None)
                else:
                    # initialize a learning curve with a placeholder
                    # This is later padded accordingly for the Conv1D layer
                    lc = [0.0]
                x_lcs.append(lc)
            self.surrogate_model.set_prediction_learning_curves(x_lcs)
            return x, inc_list
        else:
            raise ValueError(
                f"Unrecognized surrogate model name: {surrogate_name}"
            )

    def preprocess_pfn(self, x: pd.Series) -> Tuple[torch.Tensor, pd.Series, torch.Tensor]:
        """Prepares the configurations for appropriate EI calculation.

        Takes a set of points and computes the budget and incumbent for each point, as
        required by the multi-fidelity Expected Improvement acquisition function.
        """
        _x, inc_list = self.preprocess(x.copy())
        _x_tok = self.observations.tokenize(_x, as_tensor=True)
        len_partial = len(self.observations.seen_config_ids)
        z_min = x[0].fidelity.lower
        # converting fidelity to the discrete budget level
        # STRICT ASSUMPTION: fidelity is the second dimension
        _x_tok[:len_partial, 1] = (
            _x_tok[:len_partial, 1] + self.b_step - z_min
        ) / self.b_step
        return _x_tok, _x, inc_list

    def eval(self, x: pd.Series, asscalar: bool = False) -> Tuple[np.ndarray, pd.Series]:
        if self.surrogate_model_name == "pfn":
            _x_tok, _x, inc_list = self.preprocess_pfn(
                x.copy()
            )  # IMPORTANT change from vanilla-EI
            raise NotImplementedError
            # ei = self.eval_pfn_ei(_x_tok, inc_list)
        elif self.surrogate_model_name in ["deep_gp", "gp"]:
            _x, betas = self.preprocess_gp(
                x.copy(),
                self.surrogate_model_name
            )  # IMPORTANT change from vanilla-EI
            ucb = super().eval(_x.values.tolist(), betas, asscalar)
        else:
            raise ValueError(
                f"Unrecognized surrogate model name: {self.surrogate_model_name}"
            )

        return ucb, _x
