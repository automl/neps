from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import numpy as np
import pandas as pd

from neps.optimizers.base_optimizer import SampledConfig
from neps.optimizers.multi_fidelity.mf_bo import MFBOBase
from neps.optimizers.multi_fidelity.promotion_policy import AsyncPromotionPolicy
from neps.optimizers.multi_fidelity.sampling_policy import EnsemblePolicy, ModelPolicy
from neps.optimizers.multi_fidelity.successive_halving import (
    AsynchronousSuccessiveHalvingWithPriors,
)
from neps.optimizers.multi_fidelity_prior.priorband import PriorBandBase
from neps.sampling.priors import Prior

if TYPE_CHECKING:
    from neps.optimizers.bayesian_optimization.acquisition_functions import (
        BaseAcquisition,
    )
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial
    from neps.utils.types import ConfigResult, RawConfig


class PriorBandAsha(MFBOBase, PriorBandBase, AsynchronousSuccessiveHalvingWithPriors):
    """Implements a PriorBand on top of ASHA."""

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
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_default_first=sample_default_first,
            sample_default_at_target=sample_default_at_target,
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
        self.sampling_args = {
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
        self.init_size = n_min + 1  # in BOHB: init_design >= N_dim + 2
        if self.modelling_type == "joint" and self.initial_design_size is not None:
            self.init_size = self.initial_design_size

        prior_dist = Prior.from_space(self.pipeline_space)
        self.model_policy = model_policy(pipeline_space=pipeline_space, prior=prior_dist)

    def get_config_and_ids(
        self,
    ) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        rung_to_promote = self.is_promotable()
        rung = rung_to_promote + 1 if rung_to_promote is not None else self.min_rung
        self._set_sampling_weights_and_inc(rung=rung)
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
            self.sh_brackets[s] = AsynchronousSuccessiveHalvingWithPriors(**args)
            self.sh_brackets[s].sampling_policy = self.sampling_policy
            self.sh_brackets[s].sampling_args = self.sampling_args
            self.sh_brackets[s].model_policy = self.model_policy  # type: ignore
            self.sh_brackets[s].sample_new_config = self.sample_new_config  # type: ignore

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

    @override
    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
    ) -> SampledConfig:
        """This is basically the fit method."""
        completed: dict[str, ConfigResult] = {
            trial_id: trial.into_config_result(self.pipeline_space.from_dict)
            for trial_id, trial in trials.items()
            if trial.report is not None
        }
        pending: dict[str, SearchSpace] = {
            trial_id: self.pipeline_space.from_dict(trial.config)
            for trial_id, trial in trials.items()
            if trial.report is None
        }

        self.rung_histories = {
            rung: {"config": [], "perf": []}
            for rung in range(self.min_rung, self.max_rung + 1)
        }

        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))

        # previous optimization run exists and needs to be loaded
        self._load_previous_observations(completed)

        # account for pending evaluations
        self._handle_pending_evaluations(pending)

        # process optimization state and bucket observations per rung
        self._get_rungs_state()

        # filter/reset old SH brackets
        self.clear_old_brackets()

        # identifying promotion list per rung
        self._handle_promotions()

        # fit any model/surrogates
        self._fit_models()

        # important for the global HB to run the right SH
        self._update_sh_bracket_state()

        config, _id, previous_id = self.get_config_and_ids()
        return SampledConfig(id=_id, config=config, previous_config_id=previous_id)

    def _get_bracket_to_run(self) -> int:
        """Samples the ASHA bracket to run.

        The selected bracket always samples at its minimum rung. Thus, selecting a bracket
        effectively selects the rung that a new sample will be evaluated at.
        """
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
        return int(np.random.choice(range(self.max_rung + 1), p=bracket_probs))

    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # the rung to sample at
        bracket_to_run = self._get_bracket_to_run()

        self._set_sampling_weights_and_inc(rung=bracket_to_run)
        self.sh_brackets[bracket_to_run].sampling_args = self.sampling_args
        config, config_id, previous_config_id = self.sh_brackets[
            bracket_to_run
        ].get_config_and_ids()
        return config, config_id, previous_config_id
