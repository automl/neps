from __future__ import annotations

import typing

import numpy as np
from typing_extensions import Literal

from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.promotion_policy import SyncPromotionPolicy
from ..multi_fidelity.sampling_policy import EnsemblePolicy
from ..multi_fidelity.hyperband import HyperbandCustomDefault


class PriorBand(HyperbandCustomDefault):

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = EnsemblePolicy,
        promotion_policy: typing.Any = SyncPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = True,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_default_first=sample_default_first,
        )
        self.sampling_args = {
            "inc": None,
            "weights": {
                "prior": 1,  # begin with only prior sampling
                "inc": 0,
                "random": 0,
            },
        }
        for _, sh in self.sh_brackets.items():
            sh.sampling_args = self.sampling_args

    def find_incumbent(self, rung: int = None) -> SearchSpace:
        idxs = self.observed_configs.rung.values
        # filtering by rung
        while rung is not None:
            idxs = self.observed_configs.rung.values == rung
            # checking width of current rung
            if len(idxs) < self.eta:
                if rung == self.max_rung:
                    # stop if max rung reached
                    rung = None
                else:
                    # continue to next higher rung if current rung not wide enough
                    rung = rung + 1
        # extracting the incumbent configuration
        if len(idxs):
            # finding the config with the lowest recorded performance
            inc_idx = np.nanargmin(self.observed_configs.loc[idxs].perf.values)
            inc = self.observed_configs.loc[idxs].iloc[inc_idx].config
        else:
            # THIS block should not ever execute, but for runtime anomalies, if no
            # incumbent can be extracted, the prior is treated as the incumbent
            inc = self.pipeline_space.sample_default_configuration()
        return inc

    def calc_sampling_args(self) -> typing.Dict:
        sh_bracket = self.sh_brackets[self.current_sh_bracket]
        rung_size = sh_bracket.config_map[sh_bracket.min_rung]

        if self.current_sh_bracket == 0 and len(self.observed_configs) <= rung_size:
            # no incumbent selection yet
            inc = None
        else:
            inc = self.find_incumbent()

        nincs = 0 if inc is None else self.eta
        nincs = 1 if rung_size <= self.eta else nincs
        npriors = np.floor(rung_size / self.eta)
        npriors = npriors if npriors else 1
        nrandom = rung_size - npriors - nincs
        if self.current_sh_bracket == 0 and len(self.observed_configs) < npriors:
            # Enforce only prior based samples till the required number of prior samples
            # seen at the base rung of the first ever SH bracket
            nrandom = 0
        elif self.current_sh_bracket == 0 and len(self.observed_configs) >= npriors and len(self.observed_configs) <= rung_size:
            # Enforce only random samples when the required number of prior samples
            # at the base rung of the first ever SH bracket is seen
            nrandom = 1
            npriors = 0
        if self.current_sh_bracket == len(self.sh_brackets) - 1:
            # disable random search at the max rung
            nrandom = 0
            npriors = self.eta * nincs
        # normalize weights into probabilities
        _total = npriors + nincs + nrandom
        sampling_args = {
            "inc": self.find_incumbent(),
            "weights": {
                "prior": npriors / _total,
                "inc": nincs / _total,
                "random": nrandom / _total,
            },
        }
        return sampling_args

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        self.sampling_args = self.calc_sampling_args()
        for _, sh in self.sh_brackets.items():
            sh.sampling_args = self.sampling_args
        return super().get_config_and_ids()
