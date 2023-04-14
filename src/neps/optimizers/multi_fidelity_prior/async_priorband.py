from __future__ import annotations

import typing

import numpy as np
from typing_extensions import Literal

from metahyper import ConfigResult

from ...search_spaces.search_space import SearchSpace
from ..bayesian_optimization.acquisition_functions.base_acquisition import BaseAcquisition
from ..bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)
from ..multi_fidelity.mf_bo import MFBOBase
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy
from ..multi_fidelity.sampling_policy import EnsemblePolicy, ModelPolicy
from ..multi_fidelity.successive_halving import AsynchronousSuccessiveHalvingWithPriors
from ..multi_fidelity_prior.priorband import PriorBandBase


class PriorBandAsha(MFBOBase, PriorBandBase, AsynchronousSuccessiveHalvingWithPriors):
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
        sample_default_at_target: bool = True,
        prior_weight_type: str = "geometric",  # could also be {"linear", "50-50"}
        inc_sample_type: str = "mutation",  # or {"crossover", "gaussian", "hypersphere"}
        inc_mutation_rate: float = 0.5,
        inc_mutation_std: float = 0.25,
        inc_style: str = "dynamic",  # could also be {"decay", "constant"}
        # arguments for model
        model_based: bool = False,  # crucial argument to set to allow model-search
        modelling_type: str = "joint",  # could also be {"rung"}
        initial_design_size: int = None,
        model_policy: typing.Any = ModelPolicy,
        surrogate_model: str | typing.Any = "gp",
        domain_se_kernel: str = None,
        hp_kernels: list = None,
        surrogate_model_args: dict = None,
        acquisition: str | BaseAcquisition = "EI",
        log_prior_weighted: bool = False,
        acquisition_sampler: str | AcquisitionSampler = "random",
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

        bo_args = dict(
            surrogate_model=surrogate_model,
            domain_se_kernel=domain_se_kernel,
            hp_kernels=hp_kernels,
            surrogate_model_args=surrogate_model_args,
            acquisition=acquisition,
            log_prior_weighted=log_prior_weighted,
            acquisition_sampler=acquisition_sampler,
        )
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
        self.model_policy = model_policy(pipeline_space, **bo_args)

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        rung_to_promote = self.is_promotable()
        if rung_to_promote is not None:
            rung = rung_to_promote + 1
        else:
            rung = self.min_rung
        self.set_sampling_weights_and_inc(rung=rung)
        # performs standard ASHA but sampling happens as per the EnsemblePolicy
        return super().get_config_and_ids()


class PriorBandAshaHB(PriorBandAsha):
    """Implements a PriorBand on top of ASHA-HB (Mobster)."""

    early_stopping_rate: int = 0

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = EnsemblePolicy,  # key difference to ASHA
        promotion_policy: typing.Any = AsyncPromotionPolicy,  # key difference from PB
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = True,
        sample_default_at_target: bool = True,
        prior_weight_type: str = "geometric",  # could also be {"linear", "50-50"}
        inc_sample_type: str = "mutation",  # or {"crossover", "gaussian", "hypersphere"}
        inc_mutation_rate: float = 0.5,
        inc_mutation_std: float = 0.25,
        inc_style: str = "dynamic",  # could also be {"decay", "constant"}
        # arguments for model
        model_based: bool = False,  # crucial argument to set to allow model-search
        modelling_type: str = "joint",  # could also be {"rung"}
        initial_design_size: int = None,
        model_policy: typing.Any = ModelPolicy,
        surrogate_model: str | typing.Any = "gp",
        domain_se_kernel: str = None,
        hp_kernels: list = None,
        surrogate_model_args: dict = None,
        acquisition: str | BaseAcquisition = "EI",
        log_prior_weighted: bool = False,
        acquisition_sampler: str | AcquisitionSampler = "random",
    ):
        # collecting arguments required by ASHA
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
            ignore_errors=ignore_errors,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_default_first=sample_default_first,
            sample_default_at_target=sample_default_at_target,
        )
        bo_args = dict(
            surrogate_model=surrogate_model,
            domain_se_kernel=domain_se_kernel,
            hp_kernels=hp_kernels,
            surrogate_model_args=surrogate_model_args,
            acquisition=acquisition,
            log_prior_weighted=log_prior_weighted,
            acquisition_sampler=acquisition_sampler,
        )
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
            **bo_args,
        )

        # Creating the ASHA (SH) brackets that Hyperband iterates over
        self.sh_brackets = {}
        for s in range(self.max_rung + 1):
            args.update({"early_stopping_rate": s})
            # key difference from vanilla HB where it runs synchronous SH brackets
            self.sh_brackets[s] = AsynchronousSuccessiveHalvingWithPriors(**args)
            self.sh_brackets[s].sampling_policy = self.sampling_policy
            self.sh_brackets[s].sampling_args = self.sampling_args
            self.sh_brackets[s].model_policy = self.model_policy
            self.sh_brackets[s].sample_new_config = self.sample_new_config

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
        bracket_next = np.random.choice(range(self.max_rung + 1), p=bracket_probs)
        return bracket_next

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        # the rung to sample at
        bracket_to_run = self._get_bracket_to_run()

        self.set_sampling_weights_and_inc(rung=bracket_to_run)
        self.sh_brackets[bracket_to_run].sampling_args = self.sampling_args
        config, config_id, previous_config_id = self.sh_brackets[
            bracket_to_run
        ].get_config_and_ids()
        return config, config_id, previous_config_id  # type: ignore
