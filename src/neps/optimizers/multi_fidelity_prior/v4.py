import typing
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from metahyper.api import ConfigResult

from ...search_spaces.hyperparameters.categorical import CategoricalParameter
from ...search_spaces.hyperparameters.constant import ConstantParameter
from ...search_spaces.hyperparameters.float import FloatParameter
from ...search_spaces.hyperparameters.integer import IntegerParameter
from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy, SyncPromotionPolicy
from ..multi_fidelity.sampling_policy import FixedPriorPolicy
from ..multi_fidelity.successive_halving import SuccessiveHalvingWithPriors
from ..multi_fidelity_prior.v3 import OurOptimizerV3_2

# TODO: change/update/ablate
NPRIORS = 50
NRANDOM = 50
ALPHA = 0.95


def _compute_uniform(low, high, log: bool = False):
    # u = uniform(loc=low, scale=high)
    # return np.log(u.pdf(u.rvs()) + 1e-12) if log else u.pdf(u.rvs())
    p = 1 / (np.abs(high - low))
    return np.log(p + 1e-12) if log else p


def compute_uniform_prior(config: SearchSpace, log: bool = False):
    density_value = 0.0 if log else 1.0
    for hyperparameter in config.values():
        if hyperparameter.is_fidelity or isinstance(hyperparameter, ConstantParameter):
            continue
        if isinstance(hyperparameter, (FloatParameter, IntegerParameter)):
            low, high, _ = hyperparameter._get_low_high_default()
        elif isinstance(hyperparameter, CategoricalParameter):
            low = 0
            high = len(hyperparameter.choices)
        if hasattr(hyperparameter, "has_prior") and hyperparameter.has_prior:
            if log:
                density_value += _compute_uniform(low, high, log=True)
            else:
                density_value *= _compute_uniform(low, high, log=False)
    return density_value


class OurOptimizerV4_SH(SuccessiveHalvingWithPriors):
    """Implements SH with priors where priors decay to uniform per rung level."""

    npriors = NPRIORS
    nrandom = NRANDOM
    alpha = ALPHA

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = SyncPromotionPolicy,
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
            early_stopping_rate=early_stopping_rate,
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
        )
        # keeps a count of number of function evaluations made at each rung level
        self.rung_visits: Dict[int, int] = dict()

    def _load_previous_observations(
        self, previous_results: Dict[str, ConfigResult]
    ) -> None:
        # duplicating code from SuccessiveHalving to not parse previous_results again
        self.rung_visits = {rung: 0 for rung in self.rung_map.keys()}
        for config_id, config_val in previous_results.items():
            _config, _rung = self._get_config_id_split(config_id)
            # MAIN change from SuccessiveHalving for this function overloading
            self.rung_visits[int(_rung)] += 1
            if int(_config) in self.observed_configs.index:
                # config already recorded in dataframe
                rung_recorded = self.observed_configs.at[int(_config), "rung"]
                if rung_recorded < int(_rung):
                    # config recorded for a lower rung but higher rung eval available
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = self.get_loss(
                        config_val.result
                    )
            else:
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), self.get_loss(config_val.result)]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
        return

    def sample_new_config(self, **kwargs):
        # samples uniformly random
        configs = [
            self.pipeline_space.sample(
                patience=self.patience,
                user_priors=False,
                ignore_fidelity=True,
            )
            for _ in range(self.nrandom)
        ]
        # samples from prior
        configs.extend(
            [
                self.pipeline_space.sample(
                    patience=self.patience,
                    user_priors=True,
                    ignore_fidelity=True,
                )
                for _ in range(self.npriors)
            ]
        )

        rung = kwargs["rung"] if "rung" in kwargs else self.min_rung
        # polyak averaging --- slowly going towards uniform distribution
        polyak = lambda p_x, u_x, alpha: alpha * p_x + (1 - alpha) * u_x
        scores = []
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


class OurOptimizerV4_ASHA(OurOptimizerV4_SH):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,  # key difference from SH
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
            early_stopping_rate=early_stopping_rate,
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
        )

    def clear_old_brackets(self):
        "Enforces reset at each new bracket"
        # unlike synchronous SH, the state is not reset at each rung and a configuration
        # is promoted if the rung has eta configs if it is the top performing
        # base class allows for retaining the whole optimization state
        return


class OurOptimizerV4_HB(OurOptimizerV4_SH):
    early_stopping_rate = 0

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = SyncPromotionPolicy,  # key difference from SH
        loss_value_on_error: float = None,
        cost_value_on_error: float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
    ):
        args = dict(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            early_stopping_rate=self.early_stopping_rate,
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
        )
        super().__init__(**args)
        # stores the flattened sequence of SH brackets to loop over - the HB heuristic
        # for (n,r) pairing, i.e., (num. configs, fidelity)
        self.full_rung_trace = []
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            self.sh_brackets[s] = OurOptimizerV4_SH(**args)
            self.full_rung_trace.extend([s] * len(self.sh_brackets[s].full_rung_trace))
        # book-keeping variables
        self.current_sh_bracket = None
        self.old_history_len = None

    def _get_bracket_to_run(self) -> int:
        """Retrieves the exact rung ID that is being scheduled by SH in the next call."""
        bracket = self.full_rung_trace[self._counter % len(self.full_rung_trace)]
        return bracket

    def _update_state_counter(self) -> None:
        # TODO: get rid of this dependency
        self._counter += 1

    def _update_sh_bracket_state(self) -> None:
        # `load_results()` for each of the SH bracket objects are not called as they are
        # not part of the main Hyperband loop. For correct promotions and sharing of
        # optimization history, the promotion handler of the current SH bracket needs the
        # optimization state. Calling `load_results()` is an option but leads to
        # redundant data processing.
        s = self.current_sh_bracket
        self.sh_brackets[s].promotion_policy.set_state(
            max_rung=self.max_rung,
            members=self.rung_members,
            performances=self.rung_members_performance,
            config_map=self.sh_brackets[s].config_map,
        )
        self.sh_brackets[s].rung_visits = self.rung_visits
        self.sh_brackets[s].rung_promotions = self.sh_brackets[
            s
        ].promotion_policy.retrieve_promotions()
        self.sh_brackets[s].observed_configs = self.observed_configs.copy()

    def _retrieve_current_state(self) -> Tuple[int, int]:
        """Returns the current SH bracket number and the ordered index of config IDs
        that are history to ignore.
        """
        nsamples_per_bracket = sum(
            v.config_map[v.min_rung] for v in self.sh_brackets.values()
        )
        nevals_per_bracket = len(self.full_rung_trace)
        # TODO: remove dependency on `_counter` by calculating it from `observed_configs`
        nbrackets = self._counter // nevals_per_bracket
        old_history_len = nbrackets * nsamples_per_bracket
        # current HB bracket configs
        new_bracket_len = self._counter % nevals_per_bracket
        history_offset = 0
        for s in sorted(self.sh_brackets.keys()):
            _sh_bracket = self.sh_brackets[s]
            sh_bracket_len = len(_sh_bracket.full_rung_trace)
            if new_bracket_len >= sh_bracket_len:
                history_offset += _sh_bracket.config_map[_sh_bracket.min_rung]
                new_bracket_len -= sh_bracket_len
            else:
                break
        old_history_len += history_offset
        return s, old_history_len

    def clear_old_brackets(self):
        """Enforces reset at each new bracket."""
        # overloaded from SH since HB needs to not only forget the full past HB brackets
        # but also the previous SH brackets in the current HB bracket
        self.current_sh_bracket, self.old_history_len = self._retrieve_current_state()
        self.config_map = self.sh_brackets[self.current_sh_bracket].config_map
        if self.old_history_len > 0:
            # disregarding older HB brackets + older SH brackets in current HB bracket
            self._get_rungs_state(self.observed_configs.loc[self.old_history_len :])
        return

    def _handle_promotions(self):
        self.promotion_policy.set_state(
            max_rung=self.max_rung,
            members=self.rung_members,
            performances=self.rung_members_performance,
            **self.promotion_policy_kwargs,
        )
        # promotions are handled by the individual SH brackets which are explicitly
        # called in the _update_sh_bracket_state() function
        # overloaded function disables the need for retrieving promotions for HB overall
        return

    def load_results(
        self,
        previous_results: Dict[str, ConfigResult],
        pending_evaluations: Dict[str, ConfigResult],
    ) -> None:
        super().load_results(previous_results, pending_evaluations)
        # important for the global HB to run the right SH
        self._update_sh_bracket_state()

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> Tuple[SearchSpace, str, Union[str, None]]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # the current SH bracket to execute in the current HB iteration
        current_sh_bracket = self._get_bracket_to_run()
        # update SH bracket with current state of promotion candidates
        config, config_id, previous_config_id = self.sh_brackets[
            current_sh_bracket
        ].get_config_and_ids()
        # IMPORTANT function call to tell synchronous SH to query the next allocation
        self._update_state_counter()
        return config, config_id, previous_config_id  # type: ignore


class OurOptimizerV4_ASHA_HB(OurOptimizerV4_HB):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,  # key difference from SH
        loss_value_on_error: float = None,
        cost_value_on_error: float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
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
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
        )
        super().__init__(**args)
        # overwrite parent class SH brackets with Async SH brackets
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            self.sh_brackets[s] = OurOptimizerV4_ASHA(**args)

    def _update_sh_bracket_state(self) -> None:
        # `load_results()` for each of the SH bracket objects are not called as they are
        # not part of the main Hyperband loop. For correct promotions and sharing of
        # optimization history, the promotion handler of the SH brackets need the
        # optimization state. Calling `load_results()` is an option but leads to
        # redundant data processing.
        # s = self.current_sh_bracket
        for s, bracket in self.sh_brackets.items():
            bracket.promotion_policy.set_state(
                max_rung=self.max_rung,
                members=self.rung_members,
                performances=self.rung_members_performance,
                config_map=bracket.config_map,
            )
            bracket.rung_visits = self.rung_visits
            bracket.rung_promotions = bracket.promotion_policy.retrieve_promotions()
            bracket.observed_configs = self.observed_configs.copy()

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


class OurOptimizerV4_V3_2(OurOptimizerV3_2):
    """Inherits modified ASHA

    The version with modified ASHA-HB (sample a rung then promote/sample)+ the polyak
    averaging decay of the prior distribution to a uniform distribution per rung
    """

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
        # need to put stuff from OurOptimizerV4_SH for sampling and polyak decay
        # the asynchronous scheduling comes from OurOptimizerV3_2
        self.rung_visits: Dict[int, int] = dict()

    def _load_previous_observations(
        self, previous_results: Dict[str, ConfigResult]
    ) -> None:
        # duplicating code from SuccessiveHalving to not parse previous_results again
        self.rung_visits = {rung: 0 for rung in self.rung_map.keys()}
        for config_id, config_val in previous_results.items():
            _config, _rung = self._get_config_id_split(config_id)
            # only change from SuccessiveHalving
            self.rung_visits[int(_rung)] += 1
            if int(_config) in self.observed_configs.index:
                # config already recorded in dataframe
                rung_recorded = self.observed_configs.at[int(_config), "rung"]
                if rung_recorded < int(_rung):
                    # config recorded for a lower rung but higher rung eval available
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = self.get_loss(
                        config_val.result
                    )
            else:
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), self.get_loss(config_val.result)]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
        return

    def sample_new_config(self, **kwargs):
        # samples uniformly random
        configs = [
            self.pipeline_space.sample(
                patience=self.patience,
                user_priors=False,
                ignore_fidelity=True,
            )
            for _ in range(self.nrandom)
        ]
        # samples from prior
        configs.extend(
            [
                self.pipeline_space.sample(
                    patience=self.patience,
                    user_priors=True,
                    ignore_fidelity=True,
                )
                for _ in range(self.npriors)
            ]
        )

        rung = kwargs["rung"] if "rung" in kwargs else self.min_rung
        # polyak averaging --- slowly going towards uniform distribution
        polyak = lambda p_x, u_x, alpha: alpha * p_x + (1 - alpha) * u_x
        scores = []
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
