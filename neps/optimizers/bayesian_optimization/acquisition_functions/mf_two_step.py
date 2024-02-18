import numpy as np
import pandas as pd
from typing import Any, Tuple, Union

from ....search_spaces.search_space import SearchSpace
from ...multi_fidelity.utils import MFObservedData
from .base_acquisition import BaseAcquisition
from .mf_ei import MFEI, MFEI_Dyna
from .mf_ucb import MF_UCB_Dyna


class MF_TwoStep(BaseAcquisition):
    """ 2-step acquisition: employs 3 different acquisition calls.
    """

    # HYPER-PARAMETERS: Going with the Freeze-Thaw BO (Swersky et al. 2014) values
    N_PARTIAL = 10
    N_NEW = 3

    def __init__(self,
        pipeline_space: SearchSpace,
        surrogate_model_name: str = None,
        beta: float=1.0,
        maximize: bool=False,
        augmented_ei: bool = False,
        xi: float = 0.0,
        in_fill: str = "best",
        log_ei: bool = False,
    ):
        """Upper Confidence Bound (UCB) acquisition function.

        Args:
            beta: Controls the balance between exploration and exploitation.
            maximize: If True, maximize the given model, else minimize.
                DEFAULT=False, assumes minimzation.
        """
        super().__init__()
        # Acquisition 1: For trimming down partial candidate set
        self.acq_partial_filter = MFEI_Dyna_PartialFilter(  # defined below
            pipeline_space=pipeline_space,
            surrogate_model_name=surrogate_model_name,
            augmented_ei=augmented_ei,
            xi=xi,
            in_fill=in_fill,
            log_ei=log_ei
        )        
        # Acquisition 2: For trimming down new candidate set
        self.acq_new_filter = MFEI(
            pipeline_space=pipeline_space,
            surrogate_model_name=surrogate_model_name,
            augmented_ei=augmented_ei,
            xi=xi,
            in_fill=in_fill,
            log_ei=log_ei
        )
        # Acquisition 3: For final selection of winners from Acquisitions 1 & 2
        self.acq_combined = MF_UCB_Dyna(
            pipeline_space=pipeline_space,
            surrogate_model_name=surrogate_model_name,
            beta=beta,
            maximize=maximize
        )
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
        self.acq_partial_filter.set_state(
            self.pipeline_space,
            self.surrogate_model,
            self.observations,
            self.b_step
        )
        self.acq_new_filter.set_state(
            self.pipeline_space,
            self.surrogate_model,
            self.observations,
            self.b_step
        )
        self.acq_combined.set_state(
            self.pipeline_space,
            self.surrogate_model,
            self.observations,
            self.b_step
        )

    def eval(self, x: pd.Series, asscalar: bool = False) -> Tuple[np.ndarray, pd.Series]:
        # Filter self.N_NEW among the new configuration IDs
        # Filter self.N_PARTIAL among the partial configuration IDs
        max_seen_id = max(self.observations.seen_config_ids)
        total_seen_id = len(self.observations.seen_config_ids)
        new_ids = x.index[x.index > max_seen_id].values
        partial_ids = x.index[x.index <= max_seen_id].values

        # for new candidate set
        acq, _samples = self.acq_new_filter.eval(x, asscalar=True)
        acq = pd.Series(acq, index=_samples.index)
        # drop partial configurations
        acq.loc[_samples.index.values <= max_seen_id] = 0
        # NOTE: setting to 0 works as EI-based AF returns > 0
        # find configs not in top-N_NEW set as per acquisition value, to be dropped
        not_top_new_idx = acq.sort_values().index[:-self.N_NEW]  # len(acq) - N_NEW
        # drop these configurations
        acq.loc[not_top_new_idx] = 0  # to ignore in the argmax of the acquisition function
        # NOTE: setting to 0 works as EI-based AF returns > 0
        # result of first round of filtering of new candidates

        acq_new_mask = pd.Series({
            idx: val for idx, val in _samples.items() if acq.loc[idx] > 0
        })
        # for partial candidate set
        acq, _samples = self.acq_partial_filter.eval(x, asscalar=True)
        acq = pd.Series(acq, index=_samples.index)
        # weigh the acq value based on max seen for each config
        acq = self._weigh_partial_acq_scores(acq=acq)
        # drop new configurations
        acq.loc[_samples.index.values > max_seen_id] = 0  # to ignore in the argmax of the acquisition function
        # find configs not in top-N_NEW set as per acquisition value
        _top_n_partial = min(self.N_PARTIAL, total_seen_id)
        not_top_new_idx = acq.sort_values().index[:-_top_n_partial]  # acq.argsort()[::-1][_top_n_partial:]  # sorts in ascending-flips-leaves out top-N_PARTIAL
        # drop these configurations
        acq.loc[not_top_new_idx] = 0  # to ignore in the argmax of the acquisition function
        # NOTE: setting to 0 works as EI-based AF returns > 0
        # result of first round of filtering of partial candidates
        acq_partial_mask = pd.Series({
            idx: val for idx, val in _samples.items() if acq.loc[idx] > 0
        })

        eligible_set = set(
            np.concatenate([
                acq_partial_mask.index.values.tolist(),
                acq_new_mask.index.values.tolist()
            ])
        )

        # for combined selection
        acq, _samples = self.acq_combined.eval(x, asscalar=True)
        acq = pd.Series(acq, index=_samples.index)
        # applying mask from step-1 to make final selection among (N_NEW + N_PARTIAL)
        mask = acq.index.isin(eligible_set)
        # NOTE: setting to -np.inf works as MF-UCB here is max.(-LCB) instead of min.(LCB)
        acq[~mask] = -np.inf # will be ignored in the argmax of the acquisition function
        acq_combined = pd.Series({
            idx: acq.loc[idx] for idx, val in _samples.items() if acq.loc[idx] != -np.inf
        })
        # NOTE: setting to -np.inf works as MF-UCB here is max.(-LCB) instead of min.(LCB)
        acq_combined = acq_combined.reindex(acq.index, fill_value=-np.inf)
        acq = acq_combined.values
        
        return acq, _samples
    
    def _weigh_partial_acq_scores(self, acq: pd.Series) -> pd.Series:
        # find the best performance per configuration seen
        inc_list_partial = self.observations.get_best_performance_per_config()

        # removing any config indicey that have not made it till here
        _idx_drop = [_i for _i in inc_list_partial.index if _i not in acq.index]
        inc_list_partial.drop(labels=_idx_drop, inplace=True)

        # normalize the scores based on relative best seen performance per config
        _inc, _max = inc_list_partial.min(), inc_list_partial.max()
        inc_list_partial = (
            (inc_list_partial - _inc) / (_max - _inc) if _inc < _max else inc_list_partial
        )

        # calculate weights per candidate
        weights = pd.Series(1 - inc_list_partial, index=inc_list_partial.index)

        # scaling the acquisition score with weights
        acq = acq * weights

        return acq


class MFEI_PartialFilter(MFEI):
    """Custom redefinition of MF-EI with Dynamic extrapolation length to adjust incumbents.
    """

    def preprocess_inc_list(self, **kwargs) -> list:
        # the assertion exists to forcibly check the call to the super().preprocess()
        # this function overload should only affect the operation inside it
        assert "budget_list" in kwargs, "Requires the length of the candidate set."
        # we still need this as placeholder for the new candidate set
        # in this class we only work on the partial candidate set
        inc_list = super().preprocess_inc_list(budget_list=kwargs["budget_list"])

        n_partial = len(self.observations.seen_config_ids)

        # NOTE: Here we set the incumbent for EI calculation for each config to the
        # maximum it has seen, in a bid to get an expected improvement over its previous
        # observed score. This could act as a filter to diverging configurations even if
        # their overall score relative to the incumbent can be high.
        inc_list_partial = self.observations.get_best_performance_per_config()
        # updating incumbent for EI computation for the partial configs
        inc_list[:n_partial] = inc_list_partial

        return inc_list


class MFEI_Dyna_PartialFilter(MFEI_Dyna):
    """Custom redefinition of MF-EI with Dynamic extrapolation length to adjust incumbents.
    """

    def preprocess_inc_list(self, **kwargs) -> list:
        # the assertion exists to forcibly check the call to the super().preprocess()
        # this function overload should only affect the operation inside it
        assert "len_x" in kwargs, "Requires the length of the candidate set."
        # we still need this as placeholder for the new candidate set
        # in this class we only work on the partial candidate set
        inc_list = super().preprocess_inc_list(len_x=kwargs["len_x"])

        n_partial = len(self.observations.seen_config_ids)

        # NOTE: Here we set the incumbent for EI calculation for each config to the
        # maximum it has seen, in a bid to get an expected improvement over its previous
        # observed score. This could act as a filter to diverging configurations even if
        # their overall score relative to the incumbent can be high.
        inc_list_partial = self.observations.get_best_performance_per_config()

        # updating incumbent for EI computation for the partial configs
        inc_list[:n_partial] = inc_list_partial

        return inc_list
        