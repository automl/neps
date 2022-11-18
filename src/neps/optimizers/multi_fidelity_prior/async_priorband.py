from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from typing_extensions import Literal

from metahyper.api import ConfigResult

from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy
from ..multi_fidelity.sampling_policy import EnsemblePolicy
from ..multi_fidelity.successive_halving import AsynchronousSuccessiveHalvingWithPriors
from ..multi_fidelity_prior.priorband import PriorBandBase


class PriorBandAsha(PriorBandBase, AsynchronousSuccessiveHalvingWithPriors):
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
        inc_sample_type: str = "hypersphere",  # could be "gaussian" too
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
        self.inc_sample_type = inc_sample_type
        self.sampling_policy = sampling_policy(pipeline_space, self.inc_sample_type)
        self.sampling_args = {
            "inc": None,
            "weights": {
                "prior": 1,  # begin with only prior sampling
                "inc": 0,
                "random": 0,
            },
        }

    def _load_previous_observations(
        self, previous_results: dict[str, ConfigResult]
    ) -> None:
        for config_id, config_val in previous_results.items():
            _config, _rung = self._get_config_id_split(config_id)
            perf = self.get_loss(config_val.result)
            if int(_config) in self.observed_configs.index:
                # config already recorded in dataframe
                rung_recorded = self.observed_configs.at[int(_config), "rung"]
                if rung_recorded < int(_rung):
                    # config recorded for a lower rung but higher rung eval available
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = perf
            else:
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), perf]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
            # for efficiency, redefining the function to have the
            # `rung_histories` assignment inside the for loop
            self.rung_histories[int(_rung)]["config"].append(int(_config))
            self.rung_histories[int(_rung)]["perf"].append(perf)
        return

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        self.rung_histories = {
            rung: {"config": [], "perf": []}
            for rung in range(self.min_rung, self.max_rung + 1)
        }
        super().load_results(previous_results, pending_evaluations)

    def calc_sampling_args(self, rung_size, inc=None) -> dict:
        nincs = 0 if inc is None else self.eta
        nincs = 1 if rung_size <= self.eta else nincs
        npriors = np.floor(rung_size / self.eta)
        npriors = npriors if npriors else 1
        nrandom = rung_size - npriors - nincs

        if rung_size == self.config_map[self.max_rung]:
            # disable random search at the max rung
            nrandom = 0
            npriors = self.eta * nincs
        # normalize weights into probabilities
        _total = npriors + nincs + nrandom
        sampling_args = {
            "inc": inc,
            "weights": {
                "prior": npriors / _total,
                "inc": nincs / _total,
                "random": nrandom / _total,
            },
        }
        return sampling_args

    def set_sampling_weights_and_inc(self):
        # activate incumbent sampling only when an evaluation has been recorded at the
        # highest rung or the max fidelity
        activate_inc = False
        _df = self.observed_configs[self.observed_configs.rung.values == self.max_rung]
        if len(_df) > 0:
            if not np.isnan(_df.iloc[0].perf):
                activate_inc = True

        if not activate_inc:
            # only prior and random sampling while no evaluation at highest fidelity
            policy_weights = {
                "prior": 1 / 3,
                "inc": 0,
                "random": 2 / 3,
            }
            self.sampling_args = {
                "inc": None,
                "weights": policy_weights,
            }
        else:
            inc = None
            min_dist = None
            if len(self.observed_configs):
                # the `if` avoids setting an incumbent when no observation is recorded
                inc = self.find_incumbent()
                if self.inc_sample_type == "hypersphere" and inc is not None:
                    min_dist = self.find_1nn_distance_from_incumbent(inc)

            self.sampling_args = self.calc_sampling_args(
                self.config_map[self.min_rung], inc
            )
            self.sampling_args.update({"distance": min_dist})
        return self.sampling_args

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        self.set_sampling_weights_and_inc()
        # performs standard ASHA but sampling happens as per the EnsemblePolicy
        return super().get_config_and_ids()


class PriorBandAshaHB(PriorBandAsha):
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
        inc_sample_type: str = "hypersphere",  # could be "gaussian" too
    ):
        args = dict(
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
        super().__init__(**args)
        self.inc_sample_type = inc_sample_type
        self.sampling_policy = sampling_policy(pipeline_space, self.inc_sample_type)
        self.sampling_args = {
            "inc": None,
            "weights": {
                "prior": 1,  # begin with only prior sampling
                "inc": 0,
                "random": 0,
            },
        }
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            # key difference from vanilla HB where it runs synchronous SH brackets
            self.sh_brackets[s] = AsynchronousSuccessiveHalvingWithPriors(**args)
            self.sh_brackets[s].sampling_policy = self.sampling_policy
            self.sh_brackets[s].sampling_args = self.sampling_args
        self.rung_histories = None

    def _update_sh_bracket_state(self) -> None:
        # `load_results()` for each of the SH bracket objects are not called as they are
        # not part of the main Hyperband loop. For correct promotions and sharing of
        # optimization history, the promotion handler of the SH brackets need the
        # optimization state. Calling `load_results()` is an option but leads to
        # redundant data processing.
        for _, bracket in self.sh_brackets.items():
            bracket.promotion_policy.set_state(
                max_rung=self.max_rung,
                members=self.rung_members,
                performances=self.rung_members_performance,
                config_map=bracket.config_map,
            )
            bracket.rung_promotions = bracket.promotion_policy.retrieve_promotions()
            bracket.observed_configs = self.observed_configs.copy()
            bracket.rung_histories = self.rung_histories

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        super().load_results(previous_results, pending_evaluations)
        # important for the global HB to run the right SH
        self._update_sh_bracket_state()

    def _get_bracket_to_run(self):
        """Samples the ASHA bracket to run"""
        # Sampling distribution derived from Appendix A (https://arxiv.org/abs/2003.10865)
        # Adapting the distribution based on the current optimization state
        # s \in [0, max_rung] and to with the denominator's constraint, we have K > s - 1
        # and thus K \in [1, ..., max_rung, ...]
        # Since in this version, we see the full SH rung, we fix the K to max_rung
        K = self.max_rung
        bracket_probs = [
            self.eta ** (K - s) * (K + 1) / (K - s + 1) for s in range(self.max_rung + 1)
        ]
        bracket_probs = np.array(bracket_probs) / sum(bracket_probs)
        bracket_next = np.random.choice(range(self.max_rung + 1), p=bracket_probs)
        return bracket_next

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        self.set_sampling_weights_and_inc()
        bracket_to_run = self._get_bracket_to_run()
        self.sh_brackets[bracket_to_run].sampling_args = self.sampling_args
        config, config_id, previous_config_id = self.sh_brackets[
            bracket_to_run
        ].get_config_and_ids()
        return config, config_id, previous_config_id  # type: ignore
