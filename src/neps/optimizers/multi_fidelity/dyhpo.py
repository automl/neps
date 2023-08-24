# mypy: disable-error-code = assignment
from typing import Any, List, Union

import numpy as np

from metahyper import ConfigResult

from ...search_spaces.search_space import FloatParameter, IntegerParameter, SearchSpace
from ..base_optimizer import BaseOptimizer
from ..bayesian_optimization.acquisition_functions.base_acquisition import BaseAcquisition
from ..bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)
from .promotion_policy import PromotionPolicy
from .sampling_policy import (
    BaseDynamicModelPolicy,
    ModelPolicy,
    RandomPromotionDynamicPolicy,
    SamplingPolicy,
)
from .utils import MFObservedData


class DyHPOBase(BaseOptimizer):
    """
    (Under development)

    Base class for DyHPO-like algorithms.
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        step_size: Union[int, float] = 1,
        surrogate_model: Union[str, Any] = "gp",
        surrogate_model_args: dict = None,
        acquisition: Union[str, BaseAcquisition] = "EI",
        acquisition_sampler: Union[str, AcquisitionSampler] = "mutation",
        log_prior_weighted: bool = False,
        domain_se_kernel: str = None,
        optimal_assignment: bool = False,  # pylint: disable=unused-argument
        graph_kernels: list = None,
        hp_kernels: list = None,
        initial_design_size: int = 10,
        use_priors: bool = False,
        sample_default_first: bool = False,
        model_policy: Any = RandomPromotionDynamicPolicy,
        sampling_policy: Any = None,
        promotion_policy: Any = None,
        model_policy_args: Union[dict, None] = None,
        sample_policy_args: Union[dict, None] = None,
        promotion_policy_args: Union[dict, None] = None,
        promotion_type: str = "model",
        sample_type: str = "model",
        sampling_args: Union[dict, None] = None,
        loss_value_on_error: Union[None, float] = None,
        cost_value_on_error: Union[None, float] = None,
        patience: int = 100,
        ignore_errors: bool = False,
        logger=None,
    ):
        """Initialise

        Args:
            pipeline_space: Space in which to search
            budget: Maximum budget
            use_priors: Allows random samples to be generated from a default
                Samples generated from a Gaussian centered around the default value
            sampling_policy: The type of sampling procedure to use
            promotion_policy: The type of promotion procedure to use
            loss_value_on_error: Setting this and cost_value_on_error to any float will
                supress any error during bayesian optimization and will use given loss
                value instead. default: None
            cost_value_on_error: Setting this and loss_value_on_error to any float will
                supress any error during bayesian optimization and will use given cost
                value instead. default: None
            logger: logger object, or None to use the neps logger
            sample_default_first: Whether to sample the default configuration first
        """
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            patience=patience,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
            logger=logger,
        )
        self._budget_list: List[Union[int, float]] = []
        self.step_size: Union[int, float] = step_size
        self._initial_design_size = initial_design_size
        self._model_update_failed = False
        self.sample_default_first = sample_default_first

        self.promotion_type = promotion_type
        self.sample_type = sample_type
        self.sampling_args = {} if sampling_args is None else sampling_args
        self.use_priors = use_priors
        self.total_fevals: int = 0

        # TODO: Use initialized objects where possible instead of ..._args parameters.
        # This will also make it easier to write new policies for users.
        if model_policy_args is None:
            model_policy_args = dict()
        if sample_policy_args is None:
            sample_policy_args = dict()
        if promotion_policy_args is None:
            promotion_policy_args = dict()

        self.observed_configs = MFObservedData(
            columns=["config", "perf", "budget_id"],
            index_names=["config_id", "budget_id"],
        )

        if model_policy is not None:
            model_params = dict(
                pipeline_space=pipeline_space,
                surrogate_model=surrogate_model,
                domain_se_kernel=domain_se_kernel,
                hp_kernels=hp_kernels,
                graph_kernels=graph_kernels,
                surrogate_model_args=surrogate_model_args,
                acquisition=acquisition,
                use_priors=use_priors,
                log_prior_weighted=log_prior_weighted,
                acquisition_sampler=acquisition_sampler,
                logger=logger,
            )
            model_params.update(model_policy_args)
            if issubclass(model_policy, BaseDynamicModelPolicy):
                self.model_policy = model_policy(
                    observed_configs=self.observed_configs, **model_params
                )
            elif issubclass(model_policy, ModelPolicy):
                self.model_policy = model_policy(**model_params)
            elif issubclass(model_policy, SamplingPolicy):
                self.model_policy = model_policy(
                    pipeline_space=pipeline_space,
                    patience=patience,
                    logger=logger,
                    **model_policy_args,
                )
            else:
                raise ValueError(
                    f"Model policy can't be {model_policy}. "
                    f"It must subclass one of the predefined base classes"
                )

        if sampling_policy is not None:
            sampling_params = dict(
                pipeline_space=pipeline_space, patience=patience, logger=logger
            )
            if issubclass(sampling_policy, SamplingPolicy):
                sampling_params.update(sample_policy_args)
                self.sampling_policy = sampling_policy(**sampling_params)
            else:
                raise ValueError(
                    f"Sampling policy {sampling_policy} must inherit from "
                    f"SamplingPolicy base class"
                )

        if promotion_policy is not None:
            if issubclass(promotion_policy, PromotionPolicy):
                promotion_params = dict(eta=3)
                promotion_params.update(promotion_policy_args)
                self.promotion_policy = promotion_policy(**promotion_params)
            else:
                raise ValueError(
                    f"Promotion policy {promotion_policy} must inherit from "
                    f"PromotionPolicy base class"
                )

    def get_budget_level(self, config: SearchSpace) -> int:
        return int((config.fidelity.value - config.fidelity.lower) / self.step_size)

    def get_budget_value(self, budget_level: Union[int, float]) -> Union[int, float]:
        if isinstance(self.pipeline_space.fidelity, IntegerParameter):
            budget_val = int(
                self.step_size * budget_level + self.pipeline_space.fidelity.lower
            )
        elif isinstance(self.pipeline_space.fidelity, FloatParameter):
            budget_val = (
                self.step_size * budget_level + self.pipeline_space.fidelity.lower
            )
        else:
            raise NotImplementedError(
                f"Fidelity parameter: {self.pipeline_space.fidelity}"
                f"must be one of the types: "
                f"[IntegerParameter, FloatParameter], but is type:"
                f"{type(self.pipeline_space.fidelity)}"
            )
        self._budget_list.append(budget_val)
        return budget_val

    @property
    def is_init_phase(self) -> bool:
        if self.num_train_configs < self._initial_design_size:
            return True
        return False

    @property
    def num_train_configs(self):
        return len(self.observed_configs.completed_runs)

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

        # previous optimization run exists and needs to be loaded
        self._load_previous_observations(previous_results)
        self.total_fevals = len(previous_results) + len(pending_evaluations)

        # account for pending evaluations
        self._handle_pending_evaluations(pending_evaluations)

        self.observed_configs.df.sort_index(
            level=self.observed_configs.df.index.names, inplace=True
        )
        self.model_policy.observed_configs = self.observed_configs
        # fit any model/surrogates

        if not self.is_init_phase:
            self._fit_models()

    def _load_previous_observations(self, previous_results):
        for config_id, config_val in previous_results.items():
            _config, _budget_level = config_id.split("_")
            perf = self.get_loss(config_val.result)
            index = (int(_config), int(_budget_level))
            self.observed_configs.add_data(
                [config_val.config, perf, int(_budget_level)], index=index
            )

            if not np.isclose(
                self.observed_configs.df.loc[index, self.observed_configs.perf_col], perf
            ):
                self.observed_configs.update_data(
                    {
                        self.observed_configs.config_col: config_val.config,
                        self.observed_configs.perf_col: perf,
                        self.observed_configs.budget_col: int(_budget_level),
                    },
                    index=index,
                )

    def _handle_pending_evaluations(self, pending_evaluations):
        for config_id, config_val in pending_evaluations.items():
            _config, _budget_level = config_id.split("_")
            index = (int(_config), int(_budget_level))

            if index not in self.observed_configs.df.index:
                self.observed_configs.add_data(
                    [config_val.config, np.nan, int(_budget_level)], index=index
                )
            else:
                self.observed_configs.update_data(
                    {
                        self.observed_configs.config_col: config_val.config,
                        self.observed_configs.perf_col: np.nan,
                        self.observed_configs.budget_col: int(_budget_level),
                    },
                    index=index,
                )

    def _fit_models(self):
        # TODO: Once done with development catch the model update exceptions
        # and skip model based suggestions if failed (karibbov)
        self.model_policy.update_model()

    def is_promotable(self, promotion_type: str = "model") -> Union[int, None]:
        """
        Check if there are any configurations to promote, if yes then return the integer
        ID of the promoted configuration, else return None.
        """
        if promotion_type == "model":
            config_id = self.model_policy.sample(is_promotion=True, **self.sampling_args)
        elif promotion_type == "policy":
            config_id = self.promotion_policy.retrieve_promotions()
        elif promotion_type is None:
            config_id = None
        else:
            raise ValueError(
                f"'{promotion_type}' based promotion is not possible, please"
                f"use either 'model', 'policy' or None as promotion_type"
            )

        return config_id

    def sample_new_config(
        self, sample_type: str = "model", **kwargs  # pylint: disable=unused-argument
    ) -> SearchSpace:
        """
        Sample completely new configuration that
        hasn't been observed in any fidelity before.
        Your model_policy and/or sampling_policy must satisfy this constraint
        """
        if sample_type == "model":
            config = self.model_policy.sample(**self.sampling_args)
        elif sample_type == "policy":
            config = self.sampling_policy.sample(**self.sampling_args)
        elif sample_type is None:
            config = self.pipeline_space.sample(
                patience=self.patience,
                user_priors=self.use_priors,
                ignore_fidelity=True,
            )
        else:
            raise ValueError(
                f"'{sample_type}' based sampling is not possible, please"
                f"use either 'model', 'policy' or None as sampling_type"
            )

        return config

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, Union[str, None]]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        _config_id = None
        fidelity_value_set = False
        if (
            self.num_train_configs == 0
            and self.sample_default_first
            and self.pipeline_space.has_prior
        ):
            config = self.pipeline_space.sample_default_configuration(
                patience=self.patience, ignore_fidelity=False
            )
        elif (
            (self.num_train_configs == 0 and self._initial_design_size >= 1)
            or self.is_init_phase
            or self._model_update_failed
        ):
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
        else:
            for _ in range(self.patience):
                promoted_config_id = self.is_promotable(
                    promotion_type=self.promotion_type
                )
                if (
                    promoted_config_id is not None
                    and promoted_config_id in self.observed_configs.df.index.levels[0]
                ):
                    current_budget = self.observed_configs.df.loc[
                        (promoted_config_id,)
                    ].index[-1]
                    next_budget = current_budget + 1
                    config = self.observed_configs.df.loc[
                        (promoted_config_id, current_budget),
                        self.observed_configs.config_col,
                    ]
                    if np.less_equal(
                        self.get_budget_value(next_budget), config.fidelity.upper
                    ):
                        config.fidelity.value = self.get_budget_value(next_budget)
                        _config_id = promoted_config_id
                        fidelity_value_set = True
                        break
                elif promoted_config_id is not None:
                    self.logger.warn(
                        f"Configuration ID: '{promoted_config_id}' is "
                        f"not promotable because it doesn't exist in "
                        f"the observed configuration IDs: "
                        f"{self.observed_configs.df.index.levels[0]}.\n\n"
                        f"Trying to sample again..."
                    )
                else:
                    # sample_new_config must return a completely new configuration that
                    # hasn't been observed in any fidelity before
                    config = self.sample_new_config(sample_type=self.sample_type)
                    break

                # if the returned config already observed,
                # set the fidelity to the next budget level if not max already
                # else set the fidelity to the minimum budget level
                # print(config_condition)
            else:
                config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True, ignore_fidelity=False
                )

        if not fidelity_value_set:
            config.fidelity.value = self.get_budget_value(0)

        if _config_id is None:
            _config_id = (
                self.observed_configs.df.index.get_level_values(0).max() + 1
                if len(self.observed_configs.df.index.get_level_values(0)) > 0
                else 0
            )
        config_id = f"{_config_id}_{self.get_budget_level(config)}"
        # print(self.observed_configs)
        return config.hp_values(), config_id, None
