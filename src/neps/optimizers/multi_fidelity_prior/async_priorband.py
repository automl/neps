from __future__ import annotations

import typing

import numpy as np
from typing_extensions import Literal

from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.hyperband import AsynchronousHyperbandWithPriors
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy
from ..multi_fidelity.sampling_policy import EnsemblePolicy
from ..multi_fidelity.successive_halving import AsynchronousSuccessiveHalvingWithPriors


class EnsemblingSamplingPolicies:
    def is_n_config_in_all_rungs(self, n: int = 1):
        if len(self.observed_configs) == 0:
            # no observations
            return False
        nrungs = self.max_rung - self.min_rung + 1
        if len(np.unique(self.observed_configs.rung.values)) < nrungs:
            # not all rungs have seen at least one configuration
            return False
        # will work only for ASHA since assumption made is that if higher rung
        # has eta configs evaluated, all lower rungs have >eta configs
        idxs = self.observed_configs.rung.values == self.max_rung
        if sum(idxs) < n:
            # less than n configurations seen
            return False
        elif np.isnan(self.observed_configs.loc[idxs].perf.values).sum() == 0:
            # enters scope if >= n configs seen and none of them are pending
            return True
        else:
            return False

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

    def set_policy_weights(
        self, config_map=None, min_rung=None
    ) -> dict[str, float] | None:
        if config_map is None:
            config_map = self.config_map
        if min_rung is None:
            min_rung = self.min_rung
        rung_size = config_map[min_rung]

        nincs = self.eta
        nincs = 1 if rung_size <= self.eta else nincs
        npriors = np.floor(rung_size / self.eta)
        npriors = npriors if npriors else 1
        nrandom = rung_size - npriors - nincs
        # npriors = max(np.floor(rung_size / self.eta), 1)
        # nincs = 0 if self.eta >= rung_size else self.eta
        # nincs = 1 if self.eta == rung_size else nincs
        # nrandom = rung_size - npriors - nincs
        _total = npriors + nincs + nrandom
        policy_weights = {
            "prior": npriors / _total,
            "inc": nincs / _total,
            "random": nrandom / _total,
        }
        return policy_weights


class PriorBandAsha(EnsemblingSamplingPolicies, AsynchronousSuccessiveHalvingWithPriors):
    """Implements a PriorBand on top of ASHA."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = EnsemblePolicy,  # key difference to ASHA
        promotion_policy: typing.Any = AsyncPromotionPolicy,  # key difference from SH
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
            early_stopping_rate=early_stopping_rate,
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

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # set sampling policy
        if not self.is_n_config_in_all_rungs(n=1):
            # run only prior based sampling in the beginning till
            # all rungs have at least one evaluated configuration
            policy_weights = {
                "prior": 1,
                "inc": 0,
                "random": 0,
            }
        else:
            # run algorithm using the weighted ensemble of sampling policies
            policy_weights = self.set_policy_weights()
        # set the incumbent
        inc = self.find_incumbent()
        self.sampling_args = {
            "inc": inc,
            "weights": policy_weights,
        }
        return super().get_config_and_ids()


class PriorBandAshaHB(EnsemblingSamplingPolicies, AsynchronousHyperbandWithPriors):
    """Implements a PriorBand on top of ASHA-HB (Mobster)."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = EnsemblePolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,
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

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # set sampling policy
        if not self.is_n_config_in_all_rungs(n=1):
            # run only ASHA with prior based sampling in the beginning
            # till all rungs have one evaluated configuration
            policy_weights = {
                "prior": 1,
                "inc": 0,
                "random": 0,
            }
            inc = None
            # running only the first ASHA bracket
            bracket_to_run = self.min_rung
            self.sh_brackets[bracket_to_run].sampling_args = {
                "inc": inc,
                "weights": policy_weights,
            }
            return self.sh_brackets[bracket_to_run].get_config_and_ids()
        # set the policy weights for each SH bracket every round, such that when
        # the bracket/rung to run is sampled by the parent class, the chosen object
        # has their policy weights set already
        # set the incumbent
        inc = self.find_incumbent()

        # set policy weights for all SH brackets such that when the
        # bracket/rung to run is sampled by the parent class,
        # the chosen object has their policy weights set already
        for _, sh in self.sh_brackets.items():
            policy_weights = self.set_policy_weights(
                config_map=sh.config_map, min_rung=sh.min_rung
            )
            sampling_args = {
                "inc": inc,
                "weights": policy_weights,
            }
            sh.sampling_args = sampling_args
        return super().get_config_and_ids()
