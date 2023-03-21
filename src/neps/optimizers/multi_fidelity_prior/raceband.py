from __future__ import annotations

import random
import typing
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from torch.quasirandom import SobolEngine
from typing_extensions import Literal

from metahyper.api import ConfigResult

from ...search_spaces.hyperparameters.categorical import (
    CATEGORICAL_CONFIDENCE_SCORES,
    CategoricalParameter,
)
from ...search_spaces.hyperparameters.constant import ConstantParameter
from ...search_spaces.hyperparameters.float import FLOAT_CONFIDENCE_SCORES, FloatParameter
from ...search_spaces.hyperparameters.integer import IntegerParameter, float_to_integer
from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer
from ..multi_fidelity.promotion_policy import PromotionPolicy
from ..multi_fidelity.sampling_policy import SamplingPolicy
from ..multi_fidelity.successive_halving import SuccessiveHalving

SAMPLING_TOL = 1e-2
INCREASE_SAMPLE_DIST = 1 + 1e-2


def compute_config_dist(
    config_array, other_configs_array, pipeline_space, filter_zero=True
):
    is_fidelity = np.array([hp.is_fidelity for hp in pipeline_space.values()])
    is_constant = np.array(
        [type(hp) is ConstantParameter for hp in pipeline_space.values()]
    )
    is_categorical = np.array(
        [type(hp) is CategoricalParameter for hp in pipeline_space.values()]
    )
    distance_mask = is_fidelity | is_constant
    num_categorical_options = np.array(
        [
            len(hp.choices) if isinstance(hp, CategoricalParameter) else 1
            for hp in pipeline_space.values()
        ]
    )[np.newaxis, :]

    diff = config_array - other_configs_array
    diff[:, is_categorical] = (diff[:, is_categorical] != 0).astype(float)
    distances = np.linalg.norm(
        (diff)[:, ~distance_mask] / np.sqrt(num_categorical_options)[:, ~distance_mask],
        axis=1,
    )
    if filter_zero:
        return distances[distances > 0]
    else:
        return distances


def unnormalize_sample(array, pipeline_space):
    num_categorical_options = np.array(
        [
            len(hp.choices) if isinstance(hp, CategoricalParameter) else 1
            for hp in pipeline_space.values()
        ]
    )[np.newaxis, :]

    config = pipeline_space.copy()
    for dim, hp in enumerate(config.values()):
        if hp.is_fidelity or type(hp) is ConstantParameter:
            array = np.insert(array, dim, -1.0, axis=1)

    if np.any(array[num_categorical_options != 1]):
        array = array * num_categorical_options
        array[num_categorical_options != 1] = np.floor(
            array[num_categorical_options != 1]
        )
    for dim, hp in enumerate(config.values()):
        if hp.is_fidelity:
            continue
        elif type(hp) is ConstantParameter:
            # constant parameter support
            continue
        elif type(hp) is CategoricalParameter:
            hp.value = hp.choices[array[:, dim].astype(int)[0]]
        elif type(hp) is FloatParameter:
            hp.value = hp._normalization_inv(array[:, dim][0])
        elif type(hp) is IntegerParameter:
            hp.value = int(round(hp.float_hp._normalization_inv(array[:, dim][0])))
        else:
            raise ValueError("Parameter type not supported")
    return config


class RacebandSamplingPolicy(SamplingPolicy):
    num_fidelity_parameters = 1

    def __init__(
        self,
        pipeline_space,
        eta,
        use_priors=True,
        sample_local=True,
        prior_confidence=None,
        use_sobol=False,
        use_sharp_prior=False,
        randomize_policy=False,
    ):
        super().__init__(pipeline_space=pipeline_space)
        self.eta = eta
        self.sampled_configs = []
        self.num_categorical_options = np.array(
            [
                len(hp.choices) if isinstance(hp, CategoricalParameter) else 1
                for hp in pipeline_space.values()
            ]
        )[np.newaxis, :]
        self.num_constants = np.sum(
            np.array([type(hp) is ConstantParameter for hp in pipeline_space.values()])
        ).astype(int)
        self.sobol_samples = None
        self.num_previous = 0
        self.use_priors = use_priors
        self.use_sharp_prior = use_sharp_prior
        self.sample_local = sample_local
        self.use_sobol = use_sobol
        self.prior_confidence = prior_confidence
        default_values = {
            hp: val.default if val.default is not None else val.lower
            for hp, val in self.pipeline_space.hyperparameters.items()
        }
        self.pipeline_space.set_hyperparameters_from_dict(default_values)
        self.randomize_policy = randomize_policy

    def clear(self):
        self.sampled_configs = []
        self.sobol_samples = None

    def sample(
        self, num_total, previous_configs=None, previous_values=None
    ) -> SearchSpace:
        if self.randomize_policy:
            return self._sample_randomized(
                num_total,
                previous_configs=previous_configs,
                previous_values=previous_values,
            )
        else:
            return self._sample_deterministic(
                num_total,
                previous_configs=previous_configs,
                previous_values=previous_values,
            )

    def _sample_randomized(
        self, num_total, previous_configs=[], previous_values=[]
    ) -> SearchSpace:
        num_sampled = len(self.sampled_configs)
        if num_sampled == 0:
            sample = self.pipeline_space
            self.sampled_configs.append(sample)
            return sample

        local_prob = 0 if len(previous_configs) == 0 else self.eta / num_total
        prior_prob = np.floor(num_total / self.eta) / num_total
        randint = np.random.uniform()

        if randint < local_prob and len(previous_configs) > 0 and self.sample_local:
            print("LOCAL", local_prob, randint)
            best_previous_index = np.argmin(previous_values)
            prev_configs_np = np.array(
                [
                    [
                        x_.normalized().value
                        if type(x_) is not ConstantParameter
                        else 0.0
                        for x_ in list(x.values())
                    ]
                    for x in previous_configs
                ]
            )
            best_config = prev_configs_np[best_previous_index]
            sample = self._sample_neighbors(best_config, prev_configs_np)

        elif randint < (prior_prob + local_prob) and self.use_priors:
            print("prior", prior_prob, local_prob, randint)
            if num_sampled == self.eta and num_total > self.eta:
                sample = self.pipeline_space
            else:
                sample = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True, ignore_fidelity=True
                )

        else:
            sample = self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=True
            )
        self.sampled_configs.append(sample)
        return sample

    def _sample_deterministic(
        self, num_total, previous_configs=[], previous_values=[]
    ) -> SearchSpace:
        num_sampled = len(self.sampled_configs)
        num_prior = np.floor(num_total / self.eta)
        if len(previous_configs) > 0 and self.sample_local:
            num_prior += self.eta

        num_sharp_prior = (
            np.floor(num_total / self.eta**2) if self.use_sharp_prior else 0
        )
        if num_sampled < num_sharp_prior and self.use_priors:
            self._enhance_priors(self.eta - num_sampled)
            sample = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=True
            )
        elif (
            len(previous_configs) > 0
            and self.num_previous < self.eta
            and self.sample_local
        ):
            self.num_previous += 1
            best_previous_index = np.argmin(previous_values)
            prev_configs_np = np.array(
                [
                    [
                        x_.normalized().value
                        if type(x_) is not ConstantParameter
                        else 0.0
                        for x_ in list(x.values())
                    ]
                    for x in previous_configs
                ]
            )
            best_config = prev_configs_np[best_previous_index]
            sample = self._sample_neighbors(best_config, prev_configs_np)

        elif num_sampled < num_prior and self.use_priors:
            if num_sampled == self.eta and num_total > self.eta:
                sample = self.pipeline_space
            else:
                sample = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True, ignore_fidelity=True
                )

        else:
            if self.use_sobol:
                if self.sobol_samples is None:
                    self.sobol_samples = (
                        SobolEngine(
                            dimension=len(self.pipeline_space)
                            - self.num_fidelity_parameters
                            - self.num_constants,
                            scramble=True,
                        )
                        .draw(num_total - num_sampled)
                        .detach()
                        .numpy()
                    )
                # get the next sample index from the list
                normalized_sample = self.sobol_samples[0][np.newaxis, :]
                self.sobol_samples = np.delete(self.sobol_samples, 0, axis=0)
                sample = unnormalize_sample(normalized_sample, self.pipeline_space)
            else:
                sample = self.pipeline_space.sample(
                    patience=self.patience, user_priors=False, ignore_fidelity=True
                )
        self.sampled_configs.append(sample)
        return sample

    def _sample_neighbors(self, config, initial_config_set):
        distance_to_others = compute_config_dist(
            config, initial_config_set, self.pipeline_space
        )
        max_neighbor_dist = max(np.min(distance_to_others), SAMPLING_TOL)
        close_neighbor = False

        while_ctr = 0
        while not close_neighbor:
            if while_ctr % 1000 == 0:
                max_neighbor_dist *= INCREASE_SAMPLE_DIST

            neighbor = self.pipeline_space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=False
            )

            neighbor_np = np.array(
                [
                    hp.normalized().value if type(hp) is not ConstantParameter else 0.0
                    for hp in list(neighbor.values())
                ]
            )[np.newaxis, :]
            dists = compute_config_dist(config, neighbor_np, self.pipeline_space)
            close_neighbor = (
                compute_config_dist(config, neighbor_np, self.pipeline_space)
                < max_neighbor_dist
            )[0]

        return neighbor

    def _enhance_priors(self, enhance_factor=1):
        """Only applicable when priors are given along with a confidence."""
        if not self.use_priors:
            return
        for k in self.pipeline_space.keys():
            if self.pipeline_space[k].is_fidelity:
                continue
            elif isinstance(self.pipeline_space[k], (FloatParameter, IntegerParameter)):
                confidence = (
                    FLOAT_CONFIDENCE_SCORES[self.prior_confidence] ** enhance_factor
                )
                self.pipeline_space[k].default_confidence_score = confidence
            elif isinstance(self.pipeline_space[k], CategoricalParameter):
                confidence = CATEGORICAL_CONFIDENCE_SCORES[self.prior_confidence]
                self.pipeline_space[k].default_confidence_score = (
                    confidence**enhance_factor
                )


class RacebandPromotionPolicy(PromotionPolicy):
    def __init__(self, eta, pipeline_space, promotion_type="global", **kwargs):
        super().__init__(eta, **kwargs)
        self.config_map: dict = None
        self.pipeline_space = pipeline_space
        self.already_promoted: dict = {}
        self.done = False
        self.rung_promotions = None
        # can be local, global, sparse (random)
        self.promotion_type = promotion_type

    def clear(self):
        self.already_promoted: dict = {}
        self.sampled_configs = []
        self.rung_promotions = {rung: [] for rung in list(self.config_map.keys())[:-1]}

    def set_state(
        self,
        *,  # allows only keyword args
        max_rung: int,
        members: dict,
        performances: dict,
        config_map: dict,
        configs: list,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:  # pylint: disable=unused-argument
        super().set_state(max_rung=max_rung, members=members, performances=performances)
        self.config_map = config_map
        self.sampled_configs = configs
        if self.rung_promotions is None:
            self.rung_promotions = {
                rung: [] for rung in list(self.config_map.keys())[:-1]
            }

    def retrieve_promotions(self):
        if self.promotion_type == "local":
            promotions = self._retrieve_local_promotions()
        elif self.promotion_type == "global":
            promotions = self._retrieve_global_promotions()
        elif self.promotion_type == "sparse":
            promotions = self._retrieve_sparse_promotions()
        return promotions

    def _retrieve_global_promotions(self):
        assert self.config_map is not None
        min_rung = sorted(self.config_map.keys())[0]
        if min_rung == self.max_rung:
            return self.rung_promotions
        for rung in sorted(self.config_map.keys()):
            if rung == self.max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue

            if (
                self.already_promoted.get(rung, False)
                and len(self.rung_promotions[rung]) > 0
            ):
                self.rung_promotions[rung].pop(0)

            promotion_criteria = (
                len(self.rung_members_performance[rung]) == self.config_map[rung]
            )
            if promotion_criteria:
                self.already_promoted[rung] = True
                top_k = len(self.rung_members_performance[rung]) // self.eta
                best_performing_indices = np.argsort(self.rung_members_performance[rung])[
                    0:top_k
                ]
                self.rung_promotions[rung] = self.rung_members[rung][
                    best_performing_indices
                ].tolist()

        if len(self.rung_promotions[self.max_rung - 1]) == 1:
            self.done = True

        return self.rung_promotions

    def _retrieve_sparse_promotions(self):
        assert self.config_map is not None
        min_rung = sorted(self.config_map.keys())[0]
        if min_rung == self.max_rung:
            return self.rung_promotions
        for rung in sorted(self.config_map.keys()):
            if rung == self.max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            # case 1: more configs seen than the specified num. configs at that rung
            # case 2: a lower rung is eligible for promotion as num. configs already
            #   seen but when a config is promoted, the lower rung count decrements
            if (
                self.already_promoted.get(rung, False)
                and len(self.rung_promotions[rung]) > 0
            ):
                self.rung_promotions[rung].pop(0)

            promotion_criteria = (
                len(self.rung_members_performance[rung]) == self.config_map[rung]
            )
            if promotion_criteria:
                self.already_promoted[rung] = True
                top_k = len(self.rung_members_performance[rung]) // self.eta

                remaining_rung_members = self.rung_members[rung].tolist()
                remaining_performances = self.rung_members_performance[rung].tolist()
                while len(self.rung_promotions[rung]) < top_k:
                    competitor_indices = np.random.choice(
                        len(remaining_rung_members), size=self.eta, replace=False
                    )
                    competing_performances = np.array(remaining_performances)[
                        competitor_indices
                    ]
                    comp_ranking = np.argsort(competing_performances)
                    best_idx = competitor_indices[comp_ranking[0]]
                    removed_indices = competitor_indices[comp_ranking[1:]]

                    self.rung_promotions[rung].append(remaining_rung_members[best_idx])
                    for idx in np.sort(competitor_indices)[::-1]:
                        # pops the best and the two closest from the list
                        remaining_rung_members.pop(idx)
                        remaining_performances.pop(idx)

        if len(self.rung_promotions[self.max_rung - 1]) == 1:
            self.done = True
        return self.rung_promotions

    def _retrieve_local_promotions(self):
        assert self.config_map is not None
        min_rung = sorted(self.config_map.keys())[0]
        if min_rung == self.max_rung:
            return self.rung_promotions
        for rung in sorted(self.config_map.keys()):
            if rung == self.max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            # case 1: more configs seen than the specified num. configs at that rung
            # case 2: a lower rung is eligible for promotion as num. configs already
            #   seen but when a config is promoted, the lower rung count decrements
            if (
                self.already_promoted.get(rung, False)
                and len(self.rung_promotions[rung]) > 0
            ):
                self.rung_promotions[rung].pop(0)

            promotion_criteria = (
                len(self.rung_members_performance[rung]) == self.config_map[rung]
            )
            if promotion_criteria:
                self.already_promoted[rung] = True
                top_k = len(self.rung_members_performance[rung]) // self.eta
                remaining_configs = [
                    self.sampled_configs[i]
                    for i in range(len(self.sampled_configs))
                    if i in self.rung_members[rung]
                ]
                remaining_rung_members = deepcopy(self.rung_members[rung].tolist())
                remaining_performances = deepcopy(
                    self.rung_members_performance[rung].tolist()
                )
                while len(self.rung_promotions[rung]) < top_k:
                    best_idx = np.argmin(remaining_performances)
                    remaining_configs_np = np.array(
                        [
                            [
                                x_.normalized().value
                                if type(x_) is not ConstantParameter
                                else 0.0
                                for x_ in list(x.values())
                            ]
                            for x in remaining_configs
                        ]
                    )

                    best_config_np = remaining_configs_np[best_idx]
                    dist = compute_config_dist(
                        best_config_np,
                        remaining_configs_np,
                        self.pipeline_space,
                        filter_zero=False,
                    )
                    too_close_indices = np.argsort(dist)[: self.eta]
                    self.rung_promotions[rung].append(remaining_rung_members[best_idx])
                    for idx in np.sort(too_close_indices)[::-1]:
                        # pops the best and the two closest from the list
                        remaining_configs.pop(idx)
                        remaining_rung_members.pop(idx)
                        remaining_performances.pop(idx)

        if len(self.rung_promotions[self.max_rung - 1]) == 1:
            self.done = True
        return self.rung_promotions


class RaceHalving(BaseOptimizer):
    """Implements a RaceBand procedure with a sampling and promotion policy."""

    use_priors = True

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        use_priors: bool = False,
        sampling_policy: typing.Any = RacebandSamplingPolicy,
        promotion_policy: typing.Any = RacebandPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
        sampling_kwargs=None,
        promotion_kwargs=None,
        budget_variant=False,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
        )
        if random_interleave_prob < 0 or random_interleave_prob > 1:
            raise ValueError("random_interleave_prob should be in [0.0, 1.0]")
        self.random_interleave_prob = random_interleave_prob

        self.min_budget = pipeline_space.fidelity.lower
        self.max_budget = pipeline_space.fidelity.upper
        self.eta = eta
        # SH implicitly sets early_stopping_rate to 0
        # the parameter is exposed to allow HB to call SH with different stopping rates
        self.early_stopping_rate = early_stopping_rate
        self.sampling_policy = sampling_policy(
            pipeline_space, self.eta, use_priors=use_priors, **sampling_kwargs
        )
        self.promotion_policy = promotion_policy(
            self.eta, pipeline_space, **promotion_kwargs
        )
        self.budget_variant = budget_variant
        # `max_budget_init` checks for the number of configurations that have been
        # evaluated at the target budget
        self.initial_design_type = initial_design_type
        self.use_priors = use_priors
        # check to ensure no rung ID is negative
        # equivalent to s_max in https://arxiv.org/pdf/1int(idx603.06560.pdf
        self.stopping_rate_limit = np.floor(
            np.log(self.max_budget / self.min_budget) / np.log(self.eta)
        ).astype(int)
        assert self.early_stopping_rate <= self.stopping_rate_limit
        # maps rungs to a fidelity value for an SH bracket with `early_stopping_rate`
        self.rung_map = self._get_rung_map(self.early_stopping_rate)
        self.config_map = self._get_config_map(self.early_stopping_rate)
        self.min_rung = min(list(self.rung_map.keys()))
        self.max_rung = max(list(self.rung_map.keys()))

        # placeholder args for varying promotion and sampling policies
        self.promotion_policy_kwargs = {}
        self.promotion_policy_kwargs.update({"config_map": self.config_map})
        self.sampling_args: dict = {"num_total": 0}

        self.fidelities = list(self.rung_map.values())
        # stores the observations made and the corresponding fidelity explored
        # crucial data structure used for determining promotion candidates
        self.observed_configs = pd.DataFrame(
            [], columns=("config", "rung", "perf", "batch")
        )
        # stores which configs occupy each rung at any time
        self.rung_members: dict = {}  # stores config IDs per rung
        self.rung_members_performance: dict = dict()  # performances recorded per rung
        self.rung_promotions: dict = {}  # records a promotable config per rung
        self.total_fevals = 0

        # setup SH state counter
        self._counter = 0
        self.full_rung_trace = SuccessiveHalving._get_rung_trace(
            self.rung_map, self.config_map
        )
        # prior setups
        self.prior_confidence = prior_confidence
        self._enhance_priors()
        self.old_configs = None
        self.done = False

    def clear(self):
        self.counter = 0
        self.promotion_policy.clear()
        self.sampling_policy.clear()
        self.rung_members = {k: [] for k in self.rung_map.keys()}
        self.rung_members_performance = {k: [] for k in self.rung_map.keys()}
        self.rung_promotions = {}

    def transfer_results(self, results):
        self.old_results = results

    @classmethod
    def _get_rung_trace(cls, rung_map: dict, config_map: dict) -> list[int]:
        """Lists the rung IDs in sequence of the flattened SH tree."""
        rung_trace = []
        for rung in sorted(rung_map.keys()):
            rung_trace.extend([rung] * config_map[rung])
        return rung_trace

    def get_incumbent_score(self):
        y_star = np.inf  # minimizing optimizer
        if len(self.observed_configs):
            y_star = self.observed_configs.perf.values.min()
        return y_star

    def _update_state_counter(self) -> None:
        """Updates a counter to map where in the rung trace the current SH is."""
        self._counter += 1
        self._counter %= len(self.full_rung_trace)

    def _get_rung_to_run(self) -> int:
        """Retrieves the exact rung ID that is being scheduled by SH in the next call."""
        rung = self.full_rung_trace[self._counter]
        return rung

    def _get_rung_map(self, s: int = 0) -> dict:
        """Maps rungs (0,1,...,k) to a fidelity value based on fidelity bounds, eta, s."""
        assert s <= self.stopping_rate_limit
        new_min_budget = self.min_budget * (self.eta**s)
        nrungs = (
            np.floor(np.log(self.max_budget / new_min_budget) / np.log(self.eta)).astype(
                int
            )
            + 1
        )
        _max_budget = self.max_budget
        rung_map = dict()
        for i in reversed(range(nrungs)):
            # TODO: add +s to keys and TEST
            rung_map[i + s] = (
                int(_max_budget)
                if isinstance(self.pipeline_space.fidelity, IntegerParameter)
                else _max_budget
            )
            _max_budget /= self.eta
        return rung_map

    def _get_config_map(self, s: int = 0) -> dict:
        """Maps rungs (0,1,...,k) to the number of configs for each fidelity"""
        assert s <= self.stopping_rate_limit
        new_min_budget = self.min_budget * (self.eta**s)
        nrungs = (
            np.floor(np.log(self.max_budget / new_min_budget) / np.log(self.eta)).astype(
                int
            )
            + 1
        )
        s_max = self.stopping_rate_limit + 1
        _s = self.stopping_rate_limit - s
        # L2 from Alg 1 in https://arxiv.org/pdf/1603.06560.pdf
        # _n_config = np.floor(s_max / (_s + 1)) * self.eta**_s
        if self.budget_variant:
            _n_config = self.eta**_s
        else:
            _n_config = np.floor(s_max / (_s + 1)) * self.eta**_s

        config_map = dict()
        for i in range(nrungs):
            # TODO: add +s to keys and TEST
            config_map[i + s] = int(_n_config)
            _n_config //= self.eta
        return config_map

    @classmethod
    def _get_config_id_split(cls, config_id: str) -> tuple[str, str]:
        # assumes config IDs of the format `[unique config int ID]_[int rung ID]`
        _config, _rung = config_id.split("_")
        _config = str(int(_config) % 1000)
        return _config, _rung

    # TODO: check pending
    def _load_previous_observations(
        self, previous_results: dict[str, ConfigResult]
    ) -> None:
        for config_id, config_val in previous_results.items():
            _config, _rung = self._get_config_id_split(config_id)
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

    def _handle_pending_evaluations(
        self, pending_evaluations: dict[str, ConfigResult]
    ) -> None:
        # iterates over all pending evaluations and updates the list of observed
        # configs with the rung and performance as None
        for config_id, _ in pending_evaluations.items():
            _config, _rung = self._get_config_id_split(config_id)
            if int(_config) not in self.observed_configs.index:
                _df = pd.DataFrame(
                    [[None, int(_rung), None]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
            else:
                self.observed_configs.at[int(_config), "rung"] = int(_rung)
                self.observed_configs.at[int(_config), "perf"] = None
        return

    def _get_rungs_state(self):
        # to account for incomplete evaluations from being promoted --> working on a copy
        _observed_configs = self.observed_configs.copy().dropna(inplace=False)
        # iterates over the list of explored configs and buckets them to respective
        # rungs depending on the highest fidelity it was evaluated at
        self.rung_members = {k: [] for k in self.rung_map.keys()}
        self.rung_members_performance = {k: [] for k in self.rung_map.keys()}
        for _rung in _observed_configs.rung.unique():
            self.rung_members[_rung] = _observed_configs.index[
                _observed_configs.rung == _rung
            ].values
            self.rung_members_performance[_rung] = _observed_configs.perf[
                _observed_configs.rung == _rung
            ].values
        return

    def _handle_promotions(self):
        self.promotion_policy.set_state(
            max_rung=self.max_rung,
            members=self.rung_members,
            performances=self.rung_members_performance,
            configs=self.sampling_policy.sampled_configs,
            **self.promotion_policy_kwargs,
        )
        self.rung_promotions = self.promotion_policy.retrieve_promotions()

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        """This is basically the fit method.

        Args:
            previous_results (dict[str, ConfigResult]): [description]
            pending_evaluations (dict[str, ConfigResult]): [description]
        """
        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))

        # previous optimization run exists and needs to be loaded
        self._load_previous_observations(previous_results)
        self.total_fevals = len(previous_results)
        self.done = self.total_fevals == sum(list(self.config_map.values()))

        if self.done:
            return
        # account for pending evaluations
        self._handle_pending_evaluations(pending_evaluations)

        # process optimization state and bucket observations per rung
        self._get_rungs_state()

        # identifying promotion list per rung
        self._handle_promotions()

        return

    def is_promotable(self) -> int | None:
        """Returns an int if a rung can be promoted, else a None."""
        rung_next = self._get_rung_to_run()
        rung_to_promote = rung_next - 1

        if (
            rung_to_promote >= self.min_rung
            and len(self.rung_promotions[rung_to_promote]) > 0
        ):
            return rung_to_promote
        return None

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase/model-based search.

        If `initial_design_type` is `max_budget`, the number of evaluations made at the
            target budget is compared to `initial_design_size`.
        If `initial_design_type` is `unique_configs`, the total number of unique
            configurations is compared to the `initial_design_size`.
        """
        _observed_configs = self.observed_configs.copy().dropna()
        if self.initial_design_type == "max_budget":
            val = (
                np.sum(_observed_configs.rung == self.max_rung)
                < self._initial_design_size
            )
        else:
            val = len(_observed_configs) <= self._initial_design_size
        return val

    def sample_new_config(
        self,
        **kwargs,  # pylint: disable=unused-argument
    ):
        # TODO change this to generate the triplets (or well, eta-lets)
        # Samples configuration from policy or random
        rung_next = self._get_rung_to_run()
        num_total = self.config_map[rung_next]
        if self.old_results is not None:
            prev_results = [self.get_loss(el.result) for el in self.old_results.values()]
            prev_configs = [el.config for el in self.old_results.values()]
        else:
            prev_configs = None
            prev_results = None

        self.sampling_args = {
            "num_total": num_total,
            "previous_configs": prev_configs,
            "previous_values": prev_results,
        }
        config = self.sampling_policy.sample(**self.sampling_args)
        return config

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        rung_to_promote = self.is_promotable()
        if rung_to_promote is not None:
            # promotes the first recorded promotable config in the argsort-ed rung
            row = self.observed_configs.iloc[self.rung_promotions[rung_to_promote][0]]
            config = deepcopy(row["config"])
            rung = rung_to_promote + 1
            # assigning the fidelity to evaluate the config at
            config.fidelity.value = self.rung_map[rung]
            # updating config IDs
            previous_config_id = f"{row.name}_{rung_to_promote}"
            config_id = f"{row.name}_{rung}"
        else:
            rung_id = self.min_rung
            # using random instead of np.random to be consistent with NePS BO
            if random.random() < self.random_interleave_prob:
                config = self.pipeline_space.sample(
                    patience=self.patience,
                    user_priors=False,  # sample uniformly random
                    ignore_fidelity=True,
                )
            else:
                config = self.sample_new_config(rung=rung_id)
            fidelity_value = self.rung_map[rung_id]
            config.fidelity.value = fidelity_value

            previous_config_id = None
            config_id = f"{len(self.observed_configs)}_{rung_id}"

        # IMPORTANT to tell synchronous SH to query the next allocation
        self._update_state_counter()
        return config.hp_values(), config_id, previous_config_id  # type: ignore

    def _enhance_priors(self, enhance_factor=1):
        """Only applicable when priors are given along with a confidence."""
        if not self.use_priors and self.prior_confidence is None:
            return
        for k in self.pipeline_space.keys():
            if self.pipeline_space[k].is_fidelity:
                continue
            elif isinstance(self.pipeline_space[k], (FloatParameter, IntegerParameter)):
                confidence = (
                    FLOAT_CONFIDENCE_SCORES[self.prior_confidence] ** enhance_factor
                )
                self.pipeline_space[k].default_confidence_score = confidence
            elif isinstance(self.pipeline_space[k], CategoricalParameter):
                confidence = CATEGORICAL_CONFIDENCE_SCORES[self.prior_confidence]
                self.pipeline_space[k].default_confidence_score = (
                    confidence**enhance_factor
                )


class RaceBand(RaceHalving):
    """Implements a Hyperband procedure with a sampling and promotion policy."""

    early_stopping_rate = 0
    max_possible_rung = 4

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        use_priors: bool = True,
        sampling_policy: typing.Any = RacebandSamplingPolicy,
        promotion_policy: typing.Any = RacebandPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
        total_budget=10,
        budget_variant: bool = False,
        sampling_kwargs={},
        promotion_kwargs={},
    ):
        args = dict(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            early_stopping_rate=self.early_stopping_rate,  # HB subsumes this param of SH
            initial_design_type=initial_design_type,
            use_priors=use_priors,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sampling_kwargs=sampling_kwargs,
            promotion_kwargs=promotion_kwargs,
        )
        super().__init__(**args)
        # stores the flattened sequence of SH brackets to loop over - the HB heuristic
        # for (n,r) pairing, i.e., (num. configs, fidelity)
        self.sh_brackets = {}
        self.old_bracket = 0
        self.old_configs = []
        self.full_runs = 0
        if budget_variant:
            args.update({"budget_variant": True})
            self.stopping_rate_limit = np.floor(
                np.log(self.max_budget / self.min_budget) / np.log(self.eta)
            ).astype(int)
            self.max_possible_rung = min(
                self.max_possible_rung, self.stopping_rate_limit + 1
            )
            self.budget_per_rung = self._compute_rung_budget(total_budget)
            for i, s in enumerate(self.budget_per_rung):
                args.update({"early_stopping_rate": self.max_possible_rung - s})
                self.sh_brackets[i] = RaceHalving(**args)
        else:
            args.update({"budget_variant": False})
            for s in range(self.max_rung + 1):
                args.update({"early_stopping_rate": s})
                self.sh_brackets[s] = RaceHalving(**args)
        self.current_sh_bracket = 0
        self.current_sh_bracket = 0

    def _compute_rung_budget(self, max_budget):
        consumed_budget = 0
        rung_ctr = self.max_possible_rung
        rung_budgets = []
        while consumed_budget < max_budget:
            diff = max_budget - consumed_budget
            rung_budgets.append(min(rung_ctr, diff))
            consumed_budget = sum(rung_budgets)
            rung_ctr -= 1
            if rung_ctr == 0:
                rung_ctr = self.max_possible_rung
        return rung_budgets

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        filtered_previous_results = {
            key: result
            for key, result in previous_results.items()
            if key not in self.old_configs
        }
        old_results = {
            key: result
            for key, result in previous_results.items()
            if key in self.old_configs
        }

        self.sh_brackets[self.current_sh_bracket].load_results(
            filtered_previous_results, pending_evaluations
        )
        self.sh_brackets[self.current_sh_bracket].transfer_results(old_results)
        if self.sh_brackets[self.current_sh_bracket].done:
            self.current_sh_bracket += 1
            self.old_configs.extend(list(previous_results.keys()))

            # if the old one is done, we need to re-load results for the new bracket
            if self.current_sh_bracket == len(self.sh_brackets):
                self.full_runs += 1
                for bracket in self.sh_brackets.values():
                    bracket.clear()
                    self.current_sh_bracket = 0

            self.load_results(previous_results, pending_evaluations)

            # learned_configs = self.sampling_policy.get_learned_configs()

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # the current SH bracket to execute in the current HB iteration
        # update SH bracket with current state of promotion candidates
        config, config_id, previous_config_id = self.sh_brackets[
            self.current_sh_bracket
        ].get_config_and_ids()
        idx, fid = config_id.split("_")
        # IMPORTANT to tell synchronous SH to query the next allocation
        idx = (
            int(idx) % 1000
            + (self.current_sh_bracket + 1) * 1000
            + self.full_runs * 10000
        )
        config_id = f"{idx}_{fid}"
        self._update_state_counter()
        return config, config_id, previous_config_id  # type: ignore
