# mypy: disable-error-code = assignment
# type: ignore
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from metahyper import ConfigResult, instance_from_map

from ...search_spaces.search_space import FloatParameter, IntegerParameter, SearchSpace
from ..base_optimizer import BaseOptimizer
from ..bayesian_optimization.acquisition_functions import AcquisitionMapping
from ..bayesian_optimization.acquisition_functions.base_acquisition import BaseAcquisition
from ..bayesian_optimization.acquisition_functions.prior_weighted import (
    DecayingPriorWeightedAcquisition,
)
from ..bayesian_optimization.acquisition_samplers import AcquisitionSamplerMapping
from ..bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)
from ..bayesian_optimization.kernels.get_kernels import get_kernels
from .mf_bo import MFEIModel
from .utils import MFObservedData


class MFEIBO(BaseOptimizer):
    """Base class for MF-BO algorithms that use DyHPO like acquisition and budgeting."""

    acquisition: str = "MFEI"

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        step_size: int | float = 1,
        optimal_assignment: bool = False,  # pylint: disable=unused-argument
        use_priors: bool = False,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
        # sampling_policy: Any = None,
        # promotion_policy: Any = None,
        # sample_policy_args: Union[dict, None] = None,
        # promotion_policy_args: Union[dict, None] = None,
        # promotion_type: str = "model",
        # sample_type: str = "model",
        # sampling_args: Union[dict, None] = None,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        patience: int = 100,
        ignore_errors: bool = False,
        logger=None,
        # arguments for model
        surrogate_model: str | Any = "gp",
        surrogate_model_args: dict = None,
        domain_se_kernel: str = None,
        graph_kernels: list = None,
        hp_kernels: list = None,
        acquisition: str | BaseAcquisition = acquisition,
        acquisition_sampler: str | AcquisitionSampler = "freeze-thaw",
        model_policy: Any = MFEIModel,
        log_prior_weighted: bool = False,
        initial_design_size: int = 10,
        initial_design_budget: int = 100,
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
        self._budget_list: list[int | float] = []
        self.step_size: int | float = step_size
        self.min_budget = self.pipeline_space.fidelity.lower
        # TODO: generalize this to work with real data (not benchmarks)
        self.max_budget = self.pipeline_space.fidelity.upper

        self._initial_design_size = initial_design_size
        self._initial_design_budget = initial_design_budget
        self._model_update_failed = False
        self.sample_default_first = sample_default_first
        self.sample_default_at_target = sample_default_at_target

        self.use_priors = use_priors
        self.total_fevals: int = 0

        self.observed_configs = MFObservedData(
            columns=["config", "perf"],
            index_names=["config_id", "budget_id"],
        )

        # Preparing model
        graph_kernels, hp_kernels = get_kernels(
            pipeline_space=pipeline_space,
            domain_se_kernel=domain_se_kernel,
            graph_kernels=graph_kernels,
            hp_kernels=hp_kernels,
            optimal_assignment=optimal_assignment,
        )
        self.surrogate_model_args = (
            {} if surrogate_model_args is None else surrogate_model_args
        )
        self.surrogate_model_args.update(
            dict(
                # domain_se_kernel=domain_se_kernel,
                hp_kernels=hp_kernels,
                graph_kernels=graph_kernels,
            )
        )
        if not self.surrogate_model_args["hp_kernels"]:
            raise ValueError("No kernels are provided!")
        if "vectorial_features" not in self.surrogate_model_args:
            self.surrogate_model_args[
                "vectorial_features"
            ] = pipeline_space.get_vectorial_dim()
        # The surrogate model is initalized here
        self.model_policy = model_policy(
            pipeline_space=pipeline_space,
            surrogate_model=surrogate_model,
            surrogate_model_args=self.surrogate_model_args,
        )
        self.acquisition = instance_from_map(
            AcquisitionMapping,
            acquisition,
            name="acquisition function",
        )

        self.acquisition_sampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,
            name="acquisition sampler function",
            kwargs={"patience": self.patience, "pipeline_space": self.pipeline_space},
        )

    def get_budget_level(self, config: SearchSpace) -> int:
        return int((config.fidelity.value - config.fidelity.lower) / self.step_size)

    def get_budget_value(self, budget_level: int | float) -> int | float:
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
    
    def total_budget_spent(self) -> int | float:
        """ Calculates the toal budget spent so far.

        This is calculated as a function of the fidelity range provided, that takes into 
        account the minimum budget and the step size.
        """
        if len(self.observed_configs.df) == 0:
            return 0
        _df = self.observed_configs.get_learning_curves()  
        # budgets are columns now in _df
        budget_used = 0

        for idx in _df.index:
            # finds the budget steps taken per config excluding first min_budget step
            _n = (~_df.loc[idx].isna()).sum() - 1   # budget_id starts from 0
            budget_used += self.get_budget_value(_n)
        
        return budget_used

    @property
    def is_init_phase(self, budget_based: bool=True) -> bool:
        if budget_based:
            if self.total_budget_spent() < self._initial_design_budget:
                return True
        else:
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

        # an aesthetic choice more than a functional choice
        self.observed_configs.df.sort_index(
            level=self.observed_configs.df.index.names, inplace=True
        )

        # TODO: can we do better than keeping a copy of the observed configs?
        self.model_policy.observed_configs = deepcopy(self.observed_configs)
        # fit any model/surrogates
        if not self.is_init_phase:
            self._fit_models()

    def _load_previous_observations(self, previous_results):
        for config_id, config_val in previous_results.items():
            _config, _budget_level = config_id.split("_")
            perf = self.get_loss(config_val.result)
            # TODO: do we record learning curves?
            # lcs = self.get_learning_curves(config_val.result)

            index = (int(_config), int(_budget_level))
            self.observed_configs.add_data([config_val.config, perf], index=index)

            if not np.isclose(
                self.observed_configs.df.loc[index, self.observed_configs.perf_col], perf
            ):
                self.observed_configs.update_data(
                    {
                        self.observed_configs.config_col: config_val.config,
                        self.observed_configs.perf_col: perf,
                    },
                    index=index,
                )

    def _handle_pending_evaluations(self, pending_evaluations):
        for config_id, config_val in pending_evaluations.items():
            _config, _budget_level = config_id.split("_")
            index = (int(_config), int(_budget_level))

            if index not in self.observed_configs.df.index:
                self.observed_configs.add_data([config_val, np.nan], index=index)
            else:
                self.observed_configs.update_data(
                    {
                        self.observed_configs.config_col: config_val,
                        self.observed_configs.perf_col: np.nan,
                    },
                    index=index,
                )

    def _fit_models(self):
        # TODO: Once done with development catch the model update exceptions
        # and skip model based suggestions if failed (karibbov)
        self.model_policy.update_model()
        self.acquisition.set_state(
            self.model_policy.surrogate_model, self.observed_configs, self.step_size
        )
        self.acquisition_sampler.set_state(
            self.pipeline_space, self.observed_configs, self.step_size
        )

    def _sample_init_design(self) -> tuple[SearchSpace, int]:
        """ Samples the initial design.
        
        With an unbiased coin toss (p=0.5) it decides whether to sample a new 
        configuration or continue a partial configuration, until initial_design_size 
        configurations have been sampled.
        """
        _p = np.random.uniform()  # random choice
        print("*" * 50, "\n", len(self.observed_configs.seen_config_ids), "\n", "*" * 50)
        if (
            (_p < 0.5 or len(self.observed_configs.df) == 0) and 
            len(self.observed_configs.seen_config_ids) < self._initial_design_size
        ):
            # sampling a new configuration
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
            # setting the fidelity to the minimum
            config.fidelity.value = self.min_budget
            # finding the ID of the new configuration
            _config_id = self.observed_configs.next_config_id()
        else:
            # sampling a configuration ID from the observed ones
            _config_ids = np.unique(
                self.observed_configs.df.index.get_level_values('config_id').values
            )
            _config_id = np.random.choice(_config_ids)
            # extracting the config
            config = self.observed_configs.df.loc[
                _config_id, self.observed_configs.config_col
            ].iloc[0]
            # extracting the budget level
            budget = self.observed_configs.df.loc[_config_id].index.values[-1]
            # calculating fidelity value
            new_fidelity = self.get_budget_value(budget + 1)
            # settingt the config fidelity
            config.fidelity.value = new_fidelity
        return config, _config_id

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        config_id = None
        previous_config_id = None
        if (
            (self.num_train_configs == 0 and self._initial_design_size >= 1)
            or self.is_init_phase
            or self._model_update_failed
        ):
            # config = self.pipeline_space.sample(
            #     patience=self.patience, user_priors=True, ignore_fidelity=False
            # )
            # config.fidelity.value = config.fidelity.lower
            # _config_id = self.observed_configs.next_config_id()
            config, _config_id = self._sample_init_design()
        else:
            # main call here
            samples = self.acquisition_sampler.sample()
            eis = self.acquisition.eval(  # type: ignore[attr-defined]
                x=samples.to_list(), asscalar=True
            )
            # maximizing EI
            _ids = np.argsort(eis)[-1]
            # samples should have new configs with fidelities set to as required by 
            # the acquisition sampler
            config = samples.iloc[_ids]
            _config_id = samples.index[_ids]

        if _config_id in self.observed_configs.seen_config_ids:
            config_id = f"{_config_id}_{self.get_budget_level(config)}"
            previous_config_id = f"{_config_id}_{self.get_budget_level(config) - 1}"
        else:
            config_id = f"{self.observed_configs.next_config_id()}_{self.get_budget_level(config)}"

        return config.hp_values(), config_id, previous_config_id
