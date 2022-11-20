from __future__ import annotations

import random
import typing
from copy import deepcopy

import numpy as np
import pandas as pd
from typing_extensions import Literal

from metahyper.api import ConfigResult

from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy
from ..multi_fidelity.sampling_policy import EnsemblePolicy
from ..multi_fidelity.successive_halving import AsynchronousSuccessiveHalvingWithPriors
from ..multi_fidelity_prior.priorband import PriorBandBase
from .utils import DynamicWeights


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

    def set_sampling_weights_and_inc(self, rung_size: int = None):
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
            rung_size = self.config_map[self.min_rung] if rung_size is None else rung_size
            self.sampling_args = self.calc_sampling_args(rung_size, inc)
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


class AsyncPriorBand(PriorBandAsha):
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
        args = dict(
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
        super().__init__(**args)
        self.sample_map = self.rung_map.copy()
        self.promotion_map = dict()
        resources = list(sorted(self.sample_map.values()))
        for i, r in enumerate(resources[1:], start=1):
            # collects the resources spent on promotion
            # for a configuration evaluated at rungs={1,2,...,max_rung}, the resource
            # spent could be either sample_map[rung] or promotion_map[rung]
            self.promotion_map[i] = resources[i] - resources[i - 1]
        # base rung sizes are important to calculate weights for ensemble policy sampling
        self.base_rung_sizes = dict()
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            _sh_bracket = AsynchronousSuccessiveHalvingWithPriors(**args)
            _rung_size = _sh_bracket.config_map[_sh_bracket.min_rung]
            self.base_rung_sizes[s] = _rung_size
            del _sh_bracket
        # clearing excess variables and memory
        # del self.sh_brackets

    def is_promotable(self) -> bool:
        """Returns an int if a rung can be promoted, else a None."""
        rung_promotable = False

        # # iterates starting from the highest fidelity promotable to the lowest fidelity
        for rung in reversed(range(self.min_rung, self.max_rung)):
            if len(self.rung_promotions[rung]) > 0:
                rung_promotable = True
                # stop checking when a promotable config found
                # no need to search at lower fidelities
                break
        return rung_promotable

    def get_rung_of_resource(self, resource: float | int) -> int:
        if resource in self.sample_map.values():
            rung = [k for k, v in self.sample_map.items() if v == resource][0]
        elif resource in self.promotion_map.values():
            rung = [k for k, v in self.promotion_map.items() if v == resource][0]
        else:
            raise ValueError(f"{resource} not in sample or promotion resource maps")
        return rung

    def sample_resource_to_spend(self):
        r = self.min_budget
        sigma_s = (
            27 * r + 9 * r * self.eta + r * 6 * self.eta**2 + 4 * r * self.eta**3
        )
        _eta = self.eta - 1
        sigma_p = 9 * r * _eta + 6 * r * self.eta * _eta + 4 * r * _eta * self.eta**2
        _sigma_s = sigma_s / sum([sigma_s, sigma_p])
        _sigma_p = sigma_p / sum([sigma_s, sigma_p])
        p_sample = np.exp(_sigma_s) / (np.exp(_sigma_s) + np.exp(_sigma_p))
        p_promotion = np.exp(_sigma_p) / (np.exp(_sigma_s) + np.exp(_sigma_p))

        r_promotion = {
            v: [9 / 19, 6 / 19, 4 / 19][i]
            for i, v in enumerate(sorted(self.promotion_map.values()))
        }
        r_sample = {
            v: [27 / 46, 9 / 46, 6 / 46, 4 / 46][i]
            for i, v in enumerate(sorted(self.sample_map.values()))
        }

        if self.is_promotable():
            # hierarchical sampling to allow promotions
            op_choice = np.random.choice(["promote", "sample"], p=[p_promotion, p_sample])
            resource_map = r_promotion if op_choice == "promote" else r_sample
            while True:
                resource = np.random.choice(
                    list(resource_map.keys()), p=list(resource_map.values())
                )
                if op_choice == "sample":
                    # always accept the rung selected for sampling a new config
                    break

                rung = self.get_rung_of_resource(resource)
                if len(self.rung_promotions[rung - 1]):
                    # resample till the selected rung has a config to promote
                    # this scope will always contain at least a promotion in any rung
                    # or always accept the rung selected for sampling a new config
                    break
        else:
            # sample a rung to evaluate new sample at
            resource = np.random.choice(list(r_sample.keys()), p=list(r_sample.values()))
            rung = self.get_rung_of_resource(resource)
            if len(self.observed_configs) == 0 and rung == self.max_rung:
                # recovering the mode at a random rung other than the max fidelity
                while rung == self.max_rung:
                    resource = np.random.choice(
                        list(r_sample.keys()), p=list(r_sample.values())
                    )
                    rung = self.get_rung_of_resource(resource)
        return resource

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        resource = self.sample_resource_to_spend()
        rung = self.get_rung_of_resource(resource)

        if resource in self.promotion_map.values():
            # promotion
            rung_to_promote = rung - 1
            row = self.observed_configs.iloc[self.rung_promotions[rung_to_promote][0]]
            config = deepcopy(row["config"])
            config.fidelity.value = self.rung_map[rung]
            previous_config_id = f"{row.name}_{rung_to_promote}"
            config_id = f"{row.name}_{rung}"
            print(f"Promoting {rung_to_promote} to {rung}")
        else:
            # sample
            rung_id = rung
            # using random instead of np.random to be consistent with NePS BO
            if random.random() < self.random_interleave_prob:
                config = self.pipeline_space.sample(
                    patience=self.patience,
                    user_priors=False,  # sample uniformly random
                    ignore_fidelity=True,
                )
            else:
                if (
                    self.use_priors
                    and self.sample_default_first
                    and len(self.observed_configs) == 0
                ):
                    print("Sampling THE prior")
                    config = self.pipeline_space.sample_default_configuration()
                else:
                    rung_size = self.base_rung_sizes[rung_id]
                    self.set_sampling_weights_and_inc(rung_size)
                    config = self.sample_new_config(rung=rung_id)
            fidelity_value = self.rung_map[rung_id]
            config.fidelity.value = fidelity_value

            previous_config_id = None
            config_id = f"{self._generate_new_config_id()}_{rung_id}"
            print(f"Sampling at {rung}")

        return config.hp_values(), config_id, previous_config_id  # type: ignore


class AsyncPriorBandDyna(DynamicWeights, AsyncPriorBand):
    def calc_sampling_args(self, rung_size, inc=None) -> dict:
        """Sets the sampling args for the EnsemblePolicy."""
        sampling_args = super().calc_sampling_args(rung_size, inc)
        if sampling_args["weights"]["inc"] == 0 or inc is None:
            return sampling_args

        rung_to_run = [k for k, v in self.base_rung_sizes.items() if v == rung_size][0]
        rung_history = self.rung_histories[rung_to_run]
        prior = self.pipeline_space.sample_default_configuration()
        _p_prior, _p_inc = self.prior_inc_probability_ratio(rung_history, prior, inc)
        # remaining probability mass for prior and inc
        _p = 1 - sampling_args["weights"]["random"]
        # calculating scaled probabilities
        p_prior = _p_prior * _p
        p_inc = _p_inc * _p

        sampling_args = {
            "inc": inc,
            "weights": {
                "prior": p_prior,
                "inc": p_inc,
                "random": sampling_args["weights"]["random"],
            },
        }
        return sampling_args


class TrulyAsyncHB(AsyncPriorBand):
    def set_sampling_weights_and_inc(self, rung_size: int = None):
        policy_weights = {
            "prior": 0,
            "inc": 0,
            "random": 1,
        }
        self.sampling_args = {
            "inc": None,
            "weights": policy_weights,
        }
        return self.sampling_args
