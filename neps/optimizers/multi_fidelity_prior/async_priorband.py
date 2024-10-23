from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import numpy as np
import pandas as pd

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.multi_fidelity.hyperband import sample_bracket_to_run
from neps.optimizers.multi_fidelity.mf_bo import MFBOBase
from neps.optimizers.multi_fidelity.promotion_policy import AsyncPromotionPolicy
from neps.optimizers.multi_fidelity.sampling_policy import (
    EnsemblePolicy,
    ModelPolicy,
)
from neps.optimizers.multi_fidelity.successive_halving import (
    SuccessiveHalving,
    trials_to_table,
)
from neps.optimizers.multi_fidelity_prior.priorband import PriorBandBase
from neps.sampling.priors import Prior
from neps.search_spaces import Categorical, Constant, Float, Integer, SearchSpace

if TYPE_CHECKING:
    from neps.optimizers.bayesian_optimization.acquisition_functions import (
        BaseAcquisition,
    )
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial
    from neps.utils.types import RawConfig

CUSTOM_FLOAT_CONFIDENCE_SCORES = dict(Float.DEFAULT_CONFIDENCE_SCORES)
CUSTOM_FLOAT_CONFIDENCE_SCORES.update({"ultra": 0.05})

CUSTOM_CATEGORICAL_CONFIDENCE_SCORES = dict(Categorical.DEFAULT_CONFIDENCE_SCORES)
CUSTOM_CATEGORICAL_CONFIDENCE_SCORES.update({"ultra": 8})

# We should replace this with some class
RUNG_HISTORY_TYPE = dict[int, dict[Literal["config", "perf"], list[int | float]]]


class PriorBandAsha(MFBOBase, PriorBandBase, BaseOptimizer):
    """Implements a PriorBand on top of ASHA."""

    # NOTE: `sample_new_config()` and `_fit_models()` comes from MFBOBase

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: Any = EnsemblePolicy,  # key difference to ASHA
        promotion_policy: Any = AsyncPromotionPolicy,  # key difference from SH
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = True,
        sample_default_at_target: bool = True,
        prior_weight_type: Literal["geometric", "linear", "50-50"] = "geometric",
        inc_sample_type: Literal[
            "crossover", "gaussian", "hypersphere", "mutation"
        ] = "mutation",
        inc_mutation_rate: float = 0.5,
        inc_mutation_std: float = 0.25,
        inc_style: Literal["dynamic", "constant", "decay"] = "dynamic",
        # arguments for model
        model_based: bool = False,  # crucial argument to set to allow model-search
        modelling_type: Literal["joint", "rung"] = "joint",
        initial_design_size: int | None = None,
        model_policy: Any = ModelPolicy,
        # TODO: Remove these when fixing model policy
        surrogate_model: str | Any = "gp",
        domain_se_kernel: str | None = None,
        hp_kernels: list | None = None,
        surrogate_model_args: dict | None = None,
        acquisition: str | BaseAcquisition = "EI",
        log_prior_weighted: bool = False,
        acquisition_sampler: str = "random",
    ):
        if pipeline_space.fidelity is None:
            raise ValueError(
                "Fidelity should be defined in the pipeline space for"
                f" {self.__class__.__name__} to work."
            )

        if random_interleave_prob < 0 or random_interleave_prob > 1:
            raise ValueError("random_interleave_prob should be in [0.0, 1.0]")

        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
        )
        self.random_interleave_prob = random_interleave_prob
        self.sample_default_first = sample_default_first
        self.sample_default_at_target = sample_default_at_target
        self.min_budget = pipeline_space.fidelity.lower
        self.max_budget = pipeline_space.fidelity.upper
        self.eta = eta
        self.sampling_policy = sampling_policy(pipeline_space=pipeline_space)
        self.promotion_policy = promotion_policy(self.eta)
        self.initial_design_type = initial_design_type

        # check to ensure no rung ID is negative
        # equivalent to s_max in https://arxiv.org/pdf/1603.06560.pdf
        stop_rate_limit = int(
            np.floor(np.log(self.max_budget / self.min_budget) / np.log(eta))
        )
        assert early_stopping_rate <= stop_rate_limit

        # maps rungs to a fidelity value for an SH bracket with `early_stopping_rate`
        self.rung_map = {}
        nrungs = (
            int(
                np.floor(
                    np.log(
                        self.max_budget
                        / (self.min_budget * (self.eta**early_stopping_rate))
                    )
                    / np.log(self.eta)
                )
            )
            + 1
        )
        _max_budget = self.max_budget
        budget_type = int if isinstance(self.pipeline_space.fidelity, Integer) else float
        for i in reversed(range(nrungs)):
            self.rung_map[i + early_stopping_rate] = budget_type(_max_budget)
            _max_budget /= self.eta

        self.config_map: dict[int, int] = {}
        s_max = stop_rate_limit + 1
        _s = stop_rate_limit - early_stopping_rate
        # L2 from Alg 1 in https://arxiv.org/pdf/1603.06560.pdf
        _n_config = np.floor(s_max / (_s + 1)) * self.eta**_s
        for i in range(nrungs):
            self.config_map[i + early_stopping_rate] = int(_n_config)
            _n_config //= self.eta

        self.min_rung = min(self.rung_map)
        self.max_rung = max(self.rung_map)

        # placeholder args for varying promotion and sampling policies
        self.sampling_args: dict = {}
        self.fidelities = list(self.rung_map.values())
        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))
        self.rung_members: dict = {}  # stores config IDs per rung
        self.rung_members_performance: dict = {}  # performances recorded per rung
        self.rung_promotions: dict = {}  # records a promotable config per rung

        # setup SH state counter
        self.full_rung_trace = []
        for rung in sorted(self.rung_map.keys()):
            self.full_rung_trace.extend([rung] * self.config_map[rung])

        #############################
        # Setting prior confidences #
        #############################
        # the std. dev or peakiness of distribution
        self.prior_confidence = prior_confidence
        if self.use_priors and self.prior_confidence is not None:
            for k, v in self.pipeline_space.items():
                if v.is_fidelity or isinstance(v, Constant):
                    continue
                if isinstance(v, Float | Integer):
                    confidence = CUSTOM_FLOAT_CONFIDENCE_SCORES[self.prior_confidence]
                    self.pipeline_space[k].default_confidence_score = confidence
                elif isinstance(v, Categorical):
                    confidence = CUSTOM_CATEGORICAL_CONFIDENCE_SCORES[
                        self.prior_confidence
                    ]
                    self.pipeline_space[k].default_confidence_score = confidence

        self.rung_histories: RUNG_HISTORY_TYPE = {}
        self.prior_weight_type = prior_weight_type
        self.inc_sample_type = inc_sample_type
        self.inc_mutation_rate = inc_mutation_rate
        self.inc_mutation_std = inc_mutation_std
        self.sampling_policy = sampling_policy(
            pipeline_space=pipeline_space, inc_type=self.inc_sample_type
        )
        # determines the kind of trade-off between incumbent and prior weightage
        self.inc_style = inc_style  # used by PriorBandBase
        self.model_based = model_based
        self.modelling_type = modelling_type
        self.initial_design_size = initial_design_size
        self.sampling_args = {
            "inc": None,
            "weights": {
                "prior": 1,  # begin with only prior sampling
                "inc": 0,
                "random": 0,
            },
        }

        # counting non-fidelity dimensions in search space
        ndims = sum(
            1
            for _, hp in self.pipeline_space.hyperparameters.items()
            if not hp.is_fidelity
        )
        n_min = ndims + 1
        self.init_size = n_min + 1  # in BOHB: init_design >= N_dim + 2
        if self.modelling_type == "joint" and self.initial_design_size is not None:
            self.init_size = self.initial_design_size

        prior_dist = Prior.from_space(self.pipeline_space)
        self.model_policy = model_policy(pipeline_space=pipeline_space, prior=prior_dist)

    @override
    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
    ) -> SampledConfig:
        """This is basically the fit method."""
        observed_configs, rung_histories = trials_to_table(
            trials=trials,
            space=self.pipeline_space,
            rung_range=(self.min_rung, self.max_rung),
        )
        indices_of_highest_rung_per_config = observed_configs.groupby("id")[
            "rung"
        ].idxmax()
        observed_configs = observed_configs.loc[indices_of_highest_rung_per_config]
        self.observed_configs = observed_configs
        self.rung_histories = rung_histories

        # process optimization state and bucket observations per rung
        self.rung_members = {k: [] for k in self.rung_map}
        self.rung_members_performance = {k: [] for k in self.rung_map}
        self.rung_promotions = {k: [] for k in self.rung_map}

        # Drop pending
        obs_tmp = self.observed_configs.dropna(inplace=False)

        # remove the default from being part of a Successive-Halving bracket
        if (
            self.sample_default_first
            and self.sample_default_at_target
            and 0 in obs_tmp.index.values
        ):
            obs_tmp = obs_tmp.drop(index=0)

        # iterates over the list of explored configs and buckets them to respective
        # rungs depending on the highest fidelity it was evaluated at
        for _rung in obs_tmp["rung"].unique():
            idxs = obs_tmp["rung"] == _rung
            self.rung_members[_rung] = obs_tmp.index[idxs].values
            self.rung_members_performance[_rung] = obs_tmp.perf[idxs].values

        # Async so we do not clear brackets

        # identifying promotion list per rung
        self.rung_promotions = self.promotion_policy.retrieve_promotions(
            max_rung=self.max_rung,
            rung_members=self.rung_members,
            rung_members_performance=self.rung_members_performance,
            config_map=self.config_map,
        )

        # fit any model/surrogates
        # From MFBO base
        self._fit_models()

        config, _id, previous_id = self.get_config_and_ids()
        return SampledConfig(id=_id, config=config, previous_config_id=previous_id)

    def get_config_and_ids(
        self,
    ) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        for rung in reversed(range(self.min_rung, self.max_rung)):
            if len(self.rung_promotions[rung]) > 0:
                rung_to_promote = rung + 1
                break
        else:
            rung_to_promote = self.min_rung

        self._set_sampling_weights_and_inc(rung=rung_to_promote)
        # performs standard ASHA but sampling happens as per the EnsemblePolicy
        return super().get_config_and_ids()


class PriorBandAshaHB(PriorBandAsha):
    """Implements a PriorBand on top of ASHA-HB (Mobster)."""

    early_stopping_rate: int = 0

    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: Any = EnsemblePolicy,  # key difference to ASHA
        promotion_policy: Any = AsyncPromotionPolicy,  # key difference from PB
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = True,
        sample_default_at_target: bool = True,
        prior_weight_type: Literal["geometric", "linear", "50-50"] = "geometric",
        inc_sample_type: Literal[
            "crossover", "gaussian", "hypersphere", "mutation"
        ] = "mutation",
        inc_mutation_rate: float = 0.5,
        inc_mutation_std: float = 0.25,
        inc_style: Literal["dynamic", "constant", "decay"] = "dynamic",
        # arguments for model
        model_based: bool = False,  # crucial argument to set to allow model-search
        modelling_type: Literal["joint", "rung"] = "joint",
        initial_design_size: int | None = None,
        model_policy: Any = ModelPolicy,
        # TODO: Remove these when fixing model policy
        surrogate_model: str | Any = "gp",
        domain_se_kernel: str | None = None,
        hp_kernels: list | None = None,
        surrogate_model_args: dict | None = None,
        acquisition: str | BaseAcquisition = "EI",
        log_prior_weighted: bool = False,
        acquisition_sampler: str = "random",
    ):
        # collecting arguments required by ASHA
        args: dict[str, Any] = {
            "pipeline_space": pipeline_space,
            "budget": budget,
            "eta": eta,
            "early_stopping_rate": self.early_stopping_rate,
            "initial_design_type": initial_design_type,
            "sampling_policy": sampling_policy,
            "promotion_policy": promotion_policy,
            "loss_value_on_error": loss_value_on_error,
            "cost_value_on_error": cost_value_on_error,
            "ignore_errors": ignore_errors,
            "prior_confidence": prior_confidence,
            "random_interleave_prob": random_interleave_prob,
            "sample_default_first": sample_default_first,
            "sample_default_at_target": sample_default_at_target,
        }
        super().__init__(
            **args,
            prior_weight_type=prior_weight_type,
            inc_sample_type=inc_sample_type,
            inc_mutation_rate=inc_mutation_rate,
            inc_mutation_std=inc_mutation_std,
            inc_style=inc_style,
            model_based=model_based,
            modelling_type=modelling_type,
            initial_design_size=initial_design_size,
            model_policy=model_policy,
        )

        # Creating the ASHA (SH) brackets that Hyperband iterates over
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            # key difference from vanilla HB where it runs synchronous SH brackets
            self.sh_brackets[s] = SuccessiveHalving(use_priors=True, **args)
            self.sh_brackets[s].sampling_policy = self.sampling_policy
            self.sh_brackets[s].sampling_args = self.sampling_args
            self.sh_brackets[s].model_policy = self.model_policy  # type: ignore
            self.sh_brackets[s].sample_new_config = self.sample_new_config  # type: ignore

    @override
    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
    ) -> SampledConfig:
        """This is basically the fit method."""
        observed_configs, rung_histories = trials_to_table(
            trials=trials,
            space=self.pipeline_space,
            rung_range=(self.min_rung, self.max_rung),
        )
        indices_of_highest_rung_per_config = observed_configs.groupby("id")[
            "rung"
        ].idxmax()
        observed_configs = observed_configs.loc[indices_of_highest_rung_per_config]
        self.observed_configs = observed_configs
        self.rung_histories = rung_histories

        # process optimization state and bucket observations per rung
        self.rung_members = {k: [] for k in self.rung_map}
        self.rung_members_performance = {k: [] for k in self.rung_map}
        self.rung_promotions = {k: [] for k in self.rung_map}
        obs_tmp = self.observed_configs.dropna(inplace=False)

        # iterates over the list of explored configs and buckets them to respective
        # rungs depending on the highest fidelity it was evaluated at
        self.rung_members = {k: [] for k in self.rung_map}
        self.rung_members_performance = {k: [] for k in self.rung_map}
        for _rung in obs_tmp["rung"].unique():
            idxs = obs_tmp["rung"] == _rung
            self.rung_members[_rung] = obs_tmp.index[idxs].values
            self.rung_members_performance[_rung] = obs_tmp.perf[idxs].values

        # identifying promotion list per rung
        self.rung_promotions = self.promotion_policy.retrieve_promotions(
            max_rung=self.max_rung,
            rung_members=self.rung_members,
            rung_members_performance=self.rung_members_performance,
            config_map=self.config_map,
        )

        # fit any model/surrogates
        self._fit_models()

        # important for the global HB to run the right SH
        for _, bracket in self.sh_brackets.items():
            bracket.rung_promotions = bracket.promotion_policy.retrieve_promotions(
                max_rung=self.max_rung,
                rung_members=self.rung_members,
                rung_members_performance=self.rung_members_performance,
                config_map=bracket.config_map,
            )
            bracket.observed_configs = self.observed_configs
            bracket.rung_histories = self.rung_histories

        config, _id, previous_id = self.get_config_and_ids()
        return SampledConfig(id=_id, config=config, previous_config_id=previous_id)

    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # the rung to sample at
        bracket_to_run = sample_bracket_to_run(max_rung=self.max_rung, eta=self.eta)

        self._set_sampling_weights_and_inc(rung=bracket_to_run)
        self.sh_brackets[bracket_to_run].sampling_args = self.sampling_args
        config, config_id, previous_config_id = self.sh_brackets[
            bracket_to_run
        ].get_config_and_ids()
        return config, config_id, previous_config_id
