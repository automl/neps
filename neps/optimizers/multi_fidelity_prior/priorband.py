from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from neps.optimizers.multi_fidelity.hyperband import HyperbandCustomDefault
from neps.optimizers.multi_fidelity.mf_bo import MFBOBase
from neps.optimizers.multi_fidelity.promotion_policy import SyncPromotionPolicy
from neps.optimizers.multi_fidelity.sampling_policy import EnsemblePolicy, ModelPolicy
from neps.optimizers.multi_fidelity.successive_halving import SuccessiveHalvingBase
from neps.optimizers.multi_fidelity_prior.utils import (
    compute_config_dist,
    compute_scores,
    get_prior_weight_for_decay,
)
from neps.sampling.priors import Prior
from neps.search_spaces.search_space import SearchSpace

if TYPE_CHECKING:
    import pandas as pd

    from neps.optimizers.bayesian_optimization.acquisition_functions import (
        BaseAcquisition,
    )
    from neps.utils.types import RawConfig

logger = logging.getLogger(__name__)


# TODO: We should just make these functions...
class PriorBandBase:
    """Class that defines essential properties needed by PriorBand.

    Designed to work with the topmost parent class as SuccessiveHalvingBase.
    """

    # TODO: Dependant variables which should just be made into functions
    observed_configs: pd.DataFrame
    eta: int
    pipeline_space: SearchSpace
    inc_sample_type: Literal["hypersphere", "mutation", "crossover", "gaussian"]
    inc_mutation_rate: float
    inc_mutation_std: float
    rung_histories: dict[int, dict[Literal["config", "perf"], list[int | float]]]
    max_rung: int
    min_rung: int
    rung_map: dict
    prior_weight_type: Literal["geometric", "linear", "50-50"]
    sampling_args: dict[str, Any]
    inc_style: Literal["dynamic", "decay", "constant"]
    min_budget: int | float
    max_budget: int | float

    def find_all_distances_from_incumbent(self, incumbent: SearchSpace) -> list[float]:
        """Finds the distance to the nearest neighbour."""
        dist = lambda x: compute_config_dist(incumbent, x)
        # computing distance of incumbent from all seen points in history
        distances = [dist(config) for config in self.observed_configs.config]
        # ensuring the distances exclude 0 or the distance from itself
        return [d for d in distances if d > 0]

    def find_1nn_distance_from_incumbent(self, incumbent: SearchSpace) -> float:
        """Finds the distance to the nearest neighbour."""
        distances = self.find_all_distances_from_incumbent(incumbent)
        return min(distances)

    def find_incumbent(self, rung: int | None = None) -> SearchSpace:
        """Find the best performing configuration seen so far."""
        rungs = self.observed_configs.rung.values
        idxs = self.observed_configs.index.values
        while rung is not None:
            # enters this scope is `rung` argument passed and not left empty or None
            if rung not in rungs:
                logger.warning(f"{rung} not in {np.unique(idxs)}")  # type: ignore
            # filtering by rung based on argument passed
            idxs = self.observed_configs.rung.values == rung
            # checking width of current rung
            if len(idxs) < self.eta:
                logger.warn(
                    f"Selecting incumbent from a rung with width less than {self.eta}"
                )
        # extracting the incumbent configuration
        if len(idxs):
            # finding the config with the lowest recorded performance
            _perfs = self.observed_configs.loc[idxs].perf.values
            inc_idx = np.nanargmin([np.nan if t is None else t for t in _perfs])
            inc = self.observed_configs.loc[idxs].iloc[inc_idx].config
            assert isinstance(inc, SearchSpace)
        else:
            # THIS block should not ever execute, but for runtime anomalies, if no
            # incumbent can be extracted, the prior is treated as the incumbent
            inc = self.pipeline_space.from_dict(self.pipeline_space.prior_config)
            logger.warning(
                "Treating the prior as the incumbent. "
                "Please check if this should not happen."
            )
        return inc

    def _set_sampling_weights_and_inc(self, rung: int) -> dict:
        sampling_args = self.calc_sampling_args(rung)
        if not self.is_activate_inc():
            sampling_args["prior"] += sampling_args["inc"]
            sampling_args["inc"] = 0
            inc = None

            self.sampling_args = {"inc": inc, "weights": sampling_args}
        else:
            inc = self.find_incumbent()

            self.sampling_args = {"inc": inc, "weights": sampling_args}
            if self.inc_sample_type == "hypersphere":
                min_dist = self.find_1nn_distance_from_incumbent(inc)
                self.sampling_args.update({"distance": min_dist})
            elif self.inc_sample_type == "mutation":
                self.sampling_args.update(
                    {
                        "inc_mutation_rate": self.inc_mutation_rate,
                        "inc_mutation_std": self.inc_mutation_std,
                    }
                )
        return self.sampling_args

    def is_activate_inc(self) -> bool:
        """Function to check optimization state to allow/disallow incumbent sampling.

        This function checks if the total resources used for the finished evaluations
        sums to the budget of one full SH bracket.
        """
        activate_inc = False

        # calculate total resource cost required for the first SH bracket in HB
        sh_brackets = getattr(self, "sh_brackets", None)
        if sh_brackets is not None and len(sh_brackets) > 1:
            # for HB or AsyncHB which invokes multiple SH brackets
            bracket = sh_brackets[self.min_rung]
        else:
            # for SH or ASHA which do not invoke multiple SH brackets
            bracket = self

        assert isinstance(bracket, SuccessiveHalvingBase)

        # calculating the total resources spent in the first SH bracket, taking into
        # account the continuations, that is, the resources spent on a promoted config is
        # not fidelity[rung] but (fidelity[rung] - fidelity[rung - 1])
        continuation_resources = bracket.rung_map[bracket.min_rung]
        resources = bracket.config_map[bracket.min_rung] * continuation_resources
        for r in range(1, len(bracket.rung_map)):
            rung = sorted(bracket.rung_map.keys(), reverse=False)[r]
            continuation_resources = bracket.rung_map[rung] - bracket.rung_map[rung - 1]
            resources += bracket.config_map[rung] * continuation_resources

        # find resources spent so far for all finished evaluations
        valid_perf_mask = self.observed_configs["perf"].notna()
        rungs = self.observed_configs.loc[valid_perf_mask, "rung"]
        resources_used = sum(self.rung_map[r] for r in rungs)

        if resources_used >= resources and len(
            self.rung_histories[self.max_rung]["config"]
        ):
            # activate incumbent-based sampling if a total resources is at least
            # equivalent to one SH bracket resource usage, and additionally, for the
            # asynchronous case with large number of workers, the check enforces that
            # at least one configuration has been evaluated at the highest fidelity
            activate_inc = True
        return activate_inc

    def calc_sampling_args(self, rung: int) -> dict:
        """Sets the weights for each of the sampling techniques."""
        if self.prior_weight_type == "geometric":
            _w_random = 1
            # scales weight of prior by eta raised to the current rung level
            # at the base rung thus w_prior = w_random
            # at the max rung r, w_prior = eta^r * w_random
            _w_prior = (self.eta**rung) * _w_random
        elif self.prior_weight_type == "linear":
            _w_random = 1
            w_prior_min_rung = 1 * _w_random
            w_prior_max_rung = self.eta * _w_random
            num_rungs = len(self.rung_map)
            # linearly increasing prior weight such that
            # at base rung, w_prior = w_random
            # at max rung, w_prior = self.eta * w_random
            _w_prior = np.linspace(
                start=w_prior_min_rung,
                stop=w_prior_max_rung,
                endpoint=True,
                num=num_rungs,
            )[rung]
        elif self.prior_weight_type == "50-50":
            _w_random = 1
            _w_prior = 1
        else:
            raise ValueError(f"{self.prior_weight_type} not in {{'linear', 'geometric'}}")

        # normalizing weights of random and prior sampling
        w_prior = _w_prior / (_w_prior + _w_random)
        w_random = _w_random / (_w_prior + _w_random)
        # calculating ratio of prior and incumbent weights
        _w_prior, _w_inc = self.prior_to_incumbent_ratio()
        # scaling back such that w_random + w_prior + w_inc = 1
        w_inc = _w_inc * w_prior
        w_prior = _w_prior * w_prior

        return {
            "prior": w_prior,
            "inc": w_inc,
            "random": w_random,
        }

    def prior_to_incumbent_ratio(self) -> tuple[float, float]:
        """Calculates the normalized weight distribution between prior and incumbent.

        Sum of the weights should be 1.
        """
        if self.inc_style == "constant":
            return self._prior_to_incumbent_ratio_constant()
        if self.inc_style == "decay":
            valid_perf_mask = self.observed_configs["perf"].notna()
            rungs = self.observed_configs.loc[valid_perf_mask, "rung"]
            resources = sum(self.rung_map[r] for r in rungs)
            return self._prior_to_incumbent_ratio_decay(
                resources, self.eta, self.min_budget, self.max_budget
            )
        if self.inc_style == "dynamic":
            return self._prior_to_incumbent_ratio_dynamic(self.max_rung)
        raise ValueError(f"Invalid option {self.inc_style}")

    def _prior_to_incumbent_ratio_decay(
        self, resources: float, eta: int, min_budget: int | float, max_budget: int | float
    ) -> tuple[float, float]:
        """Decays the prior weightage and increases the incumbent weightage."""
        w_prior = get_prior_weight_for_decay(resources, eta, min_budget, max_budget)
        w_inc = 1 - w_prior
        return w_prior, w_inc

    def _prior_to_incumbent_ratio_constant(self) -> tuple[float, float]:
        """Fixes the weightage of incumbent sampling to 1/eta of prior sampling."""
        # fixing weight of incumbent to 1/eta of prior
        _w_prior = self.eta
        _w_inc = 1
        w_prior = _w_prior / (_w_prior + _w_inc)
        w_inc = _w_inc / (_w_prior + _w_inc)
        return w_prior, w_inc

    def _prior_to_incumbent_ratio_dynamic(self, rung: int) -> tuple[float, float]:
        """Dynamically determines the ratio of weights for prior and incumbent sampling.

        Finds the highest rung with eta configurations recorded. Picks the top-1/eta
        configs from this rung. Each config is then ranked by performance and scored by
        the Gaussian centered around the prior configuration and the Gaussian centered
        around the current incumbent. This scores each of the top-eta configs with the
        likelihood of being sampled by the prior or the incumbent. A weighted sum is
        performed on these scores based on their ranks. The ratio of the scores is used
        as the weights for prior and incumbent sampling. These weighs are calculated
        before every sampling operation.
        """
        # requires at least eta completed configurations to begin computing scores
        if len(self.rung_histories[rung]["config"]) >= self.eta:
            # retrieve the prior
            prior = self.pipeline_space.from_dict(self.pipeline_space.prior_config)
            # retrieve the global incumbent
            inc = self.find_incumbent()
            # subsetting the top 1/eta configs from the rung
            top_n = max(len(self.rung_histories[rung]["perf"]) // self.eta, self.eta)
            # ranking by performance
            config_idxs = np.argsort(self.rung_histories[rung]["perf"])[:top_n]
            # find the top-eta configurations in the rung
            top_configs = np.array(self.rung_histories[rung]["config"])[config_idxs]
            top_config_scores = np.array(
                [
                    # `compute_scores` returns a tuple of scores resp. by prior and inc
                    compute_scores(
                        self.observed_configs.loc[config_id].config, prior, inc
                    )
                    for config_id in top_configs
                ]
            )
            # adding positional weights to the score, with the best config weighed most
            weights = np.flip(np.arange(1, top_config_scores.shape[0] + 1)).reshape(-1, 1)
            # calculating weighted sum of scores
            weighted_top_config_scores = np.sum(top_config_scores * weights, axis=0)
            prior_score, inc_score = weighted_top_config_scores
            # normalizing scores to be weighted ratios
            w_prior = prior_score / sum(weighted_top_config_scores)
            w_inc = inc_score / sum(weighted_top_config_scores)
        elif rung == self.min_rung:
            # setting `w_inc = eta * w_prior` as default till score calculation begins
            w_prior = self.eta / (1 + self.eta)
            w_inc = 1 / (1 + self.eta)
        else:
            # if rung > min.rung then the lower rung could already have enough
            # configurations and thus can be recursively queried till the base rung
            return self._prior_to_incumbent_ratio_dynamic(rung - 1)
        return w_prior, w_inc


# order of inheritance (method resolution order) extremely essential for correct behaviour
class PriorBand(MFBOBase, HyperbandCustomDefault, PriorBandBase):
    """PriorBand optimizer for multi-fidelity optimization."""

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        max_cost_total: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: Any = EnsemblePolicy,
        promotion_policy: Any = SyncPromotionPolicy,
        objective_to_minimize_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
        sample_prior_first: bool = True,
        sample_prior_at_target: bool = True,
        prior_weight_type: Literal["geometric", "linear", "50-50"] = "geometric",
        inc_sample_type: Literal[
            "hypersphere", "mutation", "crossover", "gaussian"
        ] = "mutation",
        inc_mutation_rate: float = 0.5,
        inc_mutation_std: float = 0.25,
        inc_style: Literal["dynamic", "decay", "constant"] = "dynamic",
        # arguments for model
        model_based: bool = False,  # crucial argument to set to allow model-search
        modelling_type: Literal["joint", "rung"] = "joint",
        initial_design_size: int | None = None,
        model_policy: Any = ModelPolicy,
        # TODO: Remove these when fixing ModelPolicy
        surrogate_model: str | Any = "gp",
        surrogate_model_args: dict | None = None,  # TODO: Remove
        acquisition: str | BaseAcquisition = "EI",  # TODO: Remove
        log_prior_weighted: bool = False,  # TODO: Remove
        acquisition_sampler: str = "random",  # TODO: Remove
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            max_cost_total=max_cost_total,
            eta=eta,
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            objective_to_minimize_value_on_error=objective_to_minimize_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_prior_first=sample_prior_first,
            sample_prior_at_target=sample_prior_at_target,
        )
        self.prior_weight_type = prior_weight_type
        self.inc_sample_type = inc_sample_type
        self.inc_mutation_rate = inc_mutation_rate
        self.inc_mutation_std = inc_mutation_std
        self.sampling_policy = sampling_policy(
            pipeline_space=pipeline_space, inc_type=self.inc_sample_type
        )
        # determines the kind of trade-off between incumbent and prior weightage
        self.inc_style = inc_style  # used by PriorBandBase
        self.sampling_args: dict[str, Any] = {
            "inc": None,
            "weights": {
                "prior": 1,  # begin with only prior sampling
                "inc": 0,
                "random": 0,
            },
        }

        self.model_based = model_based
        self.modelling_type = modelling_type
        self.initial_design_size = initial_design_size
        # counting non-fidelity dimensions in search space
        ndims = sum(
            1
            for _, hp in self.pipeline_space.hyperparameters.items()
            if not hp.is_fidelity
        )
        n_min = ndims + 1
        self.init_size = n_min + 1  # in BOHB: init_design >= N_min + 2
        if self.modelling_type == "joint" and self.initial_design_size is not None:
            self.init_size = self.initial_design_size

        # TODO: We also create a prior later inside of `compute_scores()`,
        # in which we should really just pass in the prior dist as it does not move
        # around in the space.
        prior_dist = Prior.from_space(self.pipeline_space)
        self.model_policy = model_policy(pipeline_space=pipeline_space, prior=prior_dist)

        for _, sh in self.sh_brackets.items():
            sh.sampling_policy = self.sampling_policy
            sh.sampling_args = self.sampling_args
            sh.model_policy = self.model_policy  # type: ignore
            sh.sample_new_config = self.sample_new_config  # type: ignore

    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        self._set_sampling_weights_and_inc(rung=self.current_sh_bracket)

        for _, sh in self.sh_brackets.items():
            sh.sampling_args = self.sampling_args
        return super().get_config_and_ids()
