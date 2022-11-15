import typing
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd
from typing_extensions import Literal

from metahyper.api import ConfigResult

from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy
from ..multi_fidelity.sampling_policy import FixedPriorPolicy
from ..multi_fidelity_prior.v3 import OurOptimizerV3_2
from ..multi_fidelity_prior.v4 import ALPHA, NPRIORS, NRANDOM, compute_uniform_prior


class OurOptimizerV5(OurOptimizerV3_2):
    """Implements the property of setting the incumbent as the prior center.

    Adds only the property to set incumbent as the prior per rung to the Mod-AsyncHB v3_2.
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,
        loss_value_on_error: float = None,
        cost_value_on_error: float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
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
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
        )
        # stores the idx of the configs that have been the top eta configs at the rung
        self.rung_ranks: Dict[int, List] = dict()
        # keeps count of how many configs evaluated at each rung
        self.rung_visits: Dict[int, int] = dict()
        # number of top configs per rung to be included
        self.top_n = 1

    def _load_previous_observations(
        self, previous_results: Dict[str, ConfigResult]
    ) -> None:
        self.rung_ranks = {rung: [] for rung in self.rung_map.keys()}
        rung_rank_perfs = {rung: [] for rung in self.rung_map.keys()}
        self.rung_visits = {rung: 0 for rung in self.rung_map.keys()}
        for config_id, config_val in previous_results.items():
            _config, _rung = self._get_config_id_split(config_id)
            curr_loss = self.get_loss(config_val.result)
            self.rung_visits[int(_rung)] += 1
            self.rung_ranks[int(_rung)].append(int(_config))
            rung_rank_perfs[int(_rung)].append(curr_loss)
            if int(_config) in self.observed_configs.index:
                # config already recorded in dataframe
                rung_recorded = self.observed_configs.at[int(_config), "rung"]
                if rung_recorded < int(_rung):
                    # config recorded for a lower rung but higher rung eval available
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = curr_loss
            else:
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), curr_loss]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
        # sorting rung ranks by performance
        for rung in self.rung_map.keys():
            sorted_idxs = np.argsort(rung_rank_perfs[rung])
            self.rung_ranks[rung] = np.array(self.rung_ranks[rung])[sorted_idxs].tolist()
        return

    def _sample_from_config_as_prior(self, config_id: List[int]) -> SearchSpace:
        """Sets the prior default to be the new config and samples from new prior."""
        assert len(config_id) > 0
        # assert len(config_id) <= self.top_n
        # sampling a config from the config list
        config = deepcopy(
            np.random.choice(self.observed_configs.iloc[config_id].config.values)
        )
        # KEY STEP to set the prior by setting the default values for the NePS config
        config.set_defaults_to_current_values()
        new_config = config.sample(
            patience=self.patience,
            user_priors=True,  # hard-coding to True given the function name
            ignore_fidelity=True,
        )
        return new_config

    def select_incumbent(self, rung):
        return self.rung_ranks[rung][: self.top_n]

    def sample_new_config(
        self,
        **kwargs,  # pylint: disable=unused-argument
    ) -> SearchSpace:
        """Samples by setting the incumbent at a rung r as the prior ar rung r."""
        rung = kwargs["rung"] if "rung" in kwargs else self.min_rung
        if self.rung_visits[rung] < self.eta:
            # call ASHA's sample till every rung has seen eta configs
            return super().sample_new_config(rung=rung)

        # setting the rung's incumbent as the best ever recorded performance at the rung
        rung_incumbent = self.select_incumbent(rung)

        if self.sampling_policy is None:
            config = self.pipeline_space.sample(
                patience=self.patience,
                user_priors=self.use_priors,
                ignore_fidelity=True,
            )
        else:
            # the rung_incumbent is set to be the prior to generate a sample
            config = self._sample_from_config_as_prior(config_id=rung_incumbent)
        return config


class OurOptimizerV5_2(OurOptimizerV5):
    """Modifies V5 such that incumbent at rung+1 is the prior at rung."""

    def select_incumbent(self, rung):
        # setting the rung's incumbent as the best ever recorded performance at the next
        # higher rung, with max_rung selecting the prior at itself
        rung_for_prior = min(rung + 1, self.max_rung)
        return self.rung_ranks[rung_for_prior][: self.top_n]


class OurOptimizerV5_3(OurOptimizerV5):
    """Modifies V5 such that incumbent at rung+1 is the prior at rung."""

    def select_incumbent(self, rung):
        # TODO: use inc from rung and rung + 1
        # setting the rung's incumbent as the best ever recorded performance at the next
        # higher rung, with max_rung selecting the prior at itself
        current_rung_inc = self.rung_ranks[rung][: self.top_n]

        higher_rung_for_prior = min(rung + 1, self.max_rung)
        higher_rung_inc = []
        if len(self.rung_ranks[higher_rung_for_prior]) >= self.eta:
            higher_rung_inc = self.rung_ranks[higher_rung_for_prior][: self.top_n]

        rung_incs = current_rung_inc + higher_rung_inc
        return rung_incs


class OurOptimizerV5_V4(OurOptimizerV5):
    """Extends v5 with v4 polyak decay"""

    npriors = NPRIORS
    nrandom = NRANDOM
    alpha = ALPHA

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,
        loss_value_on_error: float = None,
        cost_value_on_error: float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
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
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
        )

    def sample_new_config(self, **kwargs):
        """Samples from a distribution averaged between the prior and uniform."""
        rung = kwargs["rung"] if "rung" in kwargs else self.min_rung

        if self.rung_visits[rung] < self.eta:
            # call ASHA's sample till every rung has seen eta configs
            return super().sample_new_config(rung=rung)

        # setting the rung's incumbent as the best ever recorded performance at the rung
        rung_incumbent = self.select_incumbent(rung)

        # setting the current prior for the rung as per v5
        prior_config = self.observed_configs.iloc[rung_incumbent[0]].config
        # KEY STEP to set the prior by setting the default values for the NePS config
        prior_config.set_defaults_to_current_values()

        # uniformly sampling the space
        configs = [
            prior_config.sample(
                patience=self.patience,
                user_priors=False,
                ignore_fidelity=True,
            )
            for _ in range(self.nrandom)
        ]
        # using the new prior to sample points
        configs.extend(
            [
                prior_config.sample(
                    patience=self.patience,
                    user_priors=True,
                    ignore_fidelity=True,
                )
                for _ in range(self.npriors)
            ]
        )

        # polyak averaging --- slowly going towards uniform distribution
        polyak = lambda p_x, u_x, alpha: alpha * p_x + (1 - alpha) * u_x
        scores = []
        # scoring each configuration based on the iteration dependent avg. distribution
        for config in configs:
            p_x = config.compute_prior()
            u_x = compute_uniform_prior(config)
            for _ in range(self.rung_visits[rung] - 1):
                p_x = polyak(p_x, u_x, self.alpha)
            scores.append(p_x)

        # sampling as a discrete distribution where the probability mass is given by the
        # normalized score of the polyak average prior score
        scores = np.array(scores) / np.sum(scores)
        config = configs[np.random.choice(range(len(configs)), p=scores)]

        return config


class OurOptimizerV5_2_V4(OurOptimizerV5_V4):
    """Extends v5_2 with v4 polyak decay"""

    def select_incumbent(self, rung):
        # setting the rung's incumbent as the best ever recorded performance at the next
        # higher rung, with max_rung selecting the prior at itself
        rung_for_prior = min(rung + 1, self.max_rung)
        return self.rung_ranks[rung_for_prior][: self.top_n]
