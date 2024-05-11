from __future__ import annotations

from typing import Any

import numpy as np

from neps.utils.types import ConfigResult, RawConfig
from neps.utils.common import instance_from_map
from neps.search_spaces.search_space import FloatParameter, IntegerParameter, SearchSpace
from neps.optimizers.base_optimizer import BaseOptimizer
from neps.optimizers.bayesian_optimization.acquisition_functions import AcquisitionMapping
from neps.optimizers.bayesian_optimization.acquisition_functions.base_acquisition import (
    BaseAcquisition,
)
from neps.optimizers.bayesian_optimization.acquisition_samplers import (
    AcquisitionSamplerMapping,
)
from neps.optimizers.bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)
from neps.optimizers.bayesian_optimization.kernels.get_kernels import get_kernels
from neps.optimizers.multi_fidelity.mf_bo import FreezeThawModel, PFNSurrogate
from neps.optimizers.multi_fidelity.utils import MFObservedData


class MFEIBO(BaseOptimizer):
    """Base class for MF-BO algorithms that use DyHPO-like acquisition and budgeting."""

    acquisition: str = "MFEI"

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int | None = None,
        step_size: int | float = 1,
        optimal_assignment: bool = False,
        use_priors: bool = False,
        sample_default_first: bool = False,
        sample_default_at_target: bool = False,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        patience: int = 100,
        ignore_errors: bool = False,
        logger=None,
        # arguments for model
        surrogate_model: str | Any = "deep_gp",
        surrogate_model_args: dict | None = None,
        domain_se_kernel: str | None = None,
        graph_kernels: list | None = None,
        hp_kernels: list | None = None,
        acquisition: str | BaseAcquisition = acquisition,
        acquisition_args: dict | None = None,
        acquisition_sampler: str | AcquisitionSampler = "freeze-thaw",
        acquisition_sampler_args: dict | None = None,
        model_policy: Any = FreezeThawModel,
        initial_design_fraction: float = 0.75,
        initial_design_size: int = 10,
        initial_design_budget: int | None = None,
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
        self.raw_tabular_space = None  # placeholder, can be populated using pre_load_hook
        self._budget_list: list[int | float] = []
        self.step_size: int | float = step_size
        self.min_budget = self.pipeline_space.fidelity.lower
        # TODO: generalize this to work with real data (not benchmarks)
        self.max_budget = self.pipeline_space.fidelity.upper

        self._initial_design_fraction = initial_design_fraction
        (
            self._initial_design_size,
            self._initial_design_budget,
        ) = self._set_initial_design(
            initial_design_size, initial_design_budget, self._initial_design_fraction
        )
        # TODO: Write use cases for these parameters
        self._model_update_failed = False
        self.sample_default_first = sample_default_first
        self.sample_default_at_target = sample_default_at_target

        self.surrogate_model_name = surrogate_model

        self.use_priors = use_priors
        self.total_fevals: int = 0

        self.observed_configs = MFObservedData(
            columns=["config", "perf", "learning_curves"],
            index_names=["config_id", "budget_id"],
        )

        # Preparing model
        self.graph_kernels, self.hp_kernels = get_kernels(
            pipeline_space=pipeline_space,
            domain_se_kernel=domain_se_kernel,
            graph_kernels=graph_kernels,
            hp_kernels=hp_kernels,
            optimal_assignment=optimal_assignment,
        )
        self.surrogate_model_args = (
            {} if surrogate_model_args is None else surrogate_model_args
        )
        self._prep_model_args(self.hp_kernels, self.graph_kernels, pipeline_space)

        # TODO: Better solution than branching based on the surrogate name is needed
        if surrogate_model in ["deep_gp", "gp"]:
            model_policy = FreezeThawModel
        elif surrogate_model == "pfn":
            model_policy = PFNSurrogate
        else:
            raise ValueError("Invalid model option selected!")

        # The surrogate model is initalized here
        self.model_policy = model_policy(
            pipeline_space=pipeline_space,
            surrogate_model=surrogate_model,
            surrogate_model_args=self.surrogate_model_args,
        )
        self.acquisition_args = {} if acquisition_args is None else acquisition_args
        self.acquisition_args.update(
            {
                "pipeline_space": self.pipeline_space,
                "surrogate_model_name": self.surrogate_model_name,
            }
        )
        self.acquisition = instance_from_map(
            AcquisitionMapping,
            acquisition,
            name="acquisition function",
            kwargs=self.acquisition_args,
        )
        self.acquisition_sampler_args = (
            {} if acquisition_sampler_args is None else acquisition_sampler_args
        )
        self.acquisition_sampler_args.update(
            {"patience": self.patience, "pipeline_space": self.pipeline_space}
        )
        self.acquisition_sampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,
            name="acquisition sampler function",
            kwargs=self.acquisition_sampler_args,
        )
        self.count = 0

    def _prep_model_args(self, hp_kernels, graph_kernels, pipeline_space):
        if self.surrogate_model_name in ["gp", "gp_hierarchy"]:
            # setup for GP implemented in NePS
            self.surrogate_model_args.update(
                dict(
                    # domain_se_kernel=domain_se_kernel,
                    hp_kernels=hp_kernels,
                    graph_kernels=graph_kernels,
                )
            )
            if not self.surrogate_model_args["hp_kernels"]:
                raise ValueError("No kernels are provided!")
            # if "vectorial_features" not in self.surrogate_model_args:
            self.surrogate_model_args["vectorial_features"] = (
                pipeline_space.raw_tabular_space.get_vectorial_dim()
                if pipeline_space.has_tabular
                else pipeline_space.get_vectorial_dim()
            )

    def _set_initial_design(
        self,
        initial_design_size: int = None,
        initial_design_budget: int = None,
        initial_design_fraction: float = 0.75,
    ) -> tuple[int | float, int | float]:
        """Sets the initial design size and budget."""

        # user specified initial_design_size takes precedence
        if initial_design_budget is not None:
            _initial_design_budget = initial_design_budget
        else:
            _initial_design_budget = self.max_budget

        # user specified initial_design_size takes precedence
        _initial_design_size = np.inf
        if initial_design_size is not None:
            _initial_design_size = initial_design_size
        if (
            initial_design_size is None
            or _initial_design_size * self.min_budget > _initial_design_budget
        ):
            # if the initial design budget is less than the budget spent on sampling
            # the initial design at the minimum budget (fidelity)
            # 2 choices here:
            #    1. Reduce initial_design_size
            #    2. Increase initial_design_budget
            # we choose to reduce initial_design_size
            _init_budget = initial_design_fraction * self.max_budget
            # number of min budget evaluations fitting within initial design budget
            _initial_design_size = _init_budget // self.min_budget

        self.logger.info(
            f"\n\ninitial_design_size: {_initial_design_size}\n"
            f"initial_design_budget: {_initial_design_budget}\n"
            f"min_budget: {self.min_budget}\n\n"
        )
        return _initial_design_size, _initial_design_budget

    def get_budget_level(self, config: SearchSpace) -> int:
        return int(
            np.ceil((config.fidelity.value - config.fidelity.lower) / self.step_size)
        )

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
        """Calculates the toal budget spent so far.

        This is calculated as a function of the fidelity range provided, that takes into
        account the minimum budget and the step size.
        """
        if len(self.observed_configs.df) == 0:
            return 0

        n_configs = len(self.observed_configs.seen_config_ids)
        total_budget_level = sum(self.observed_configs.seen_budget_levels)
        total_initial_budget_spent = n_configs * self.pipeline_space.fidelity.lower
        total_budget_spent = (
            total_initial_budget_spent + total_budget_level * self.step_size
        )

        return total_budget_spent

    def is_init_phase(self, budget_based: bool = True) -> bool:
        if budget_based:
            # Check if we are still in the initial design phase based on
            # either the budget spent so far or the number of configurations evaluated
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
        pending_evaluations: dict[str, SearchSpace],
    ) -> None:
        """This is basically the fit method.

        Args:
            previous_results (dict[str, ConfigResult]): [description]
            pending_evaluations (dict[str, ConfigResult]): [description]
        """
        self.observed_configs = MFObservedData(
            columns=["config", "perf", "learning_curves"],
            index_names=["config_id", "budget_id"],
        )

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
        # TODO: can we not hide this in load_results and have something that pops out
        #   more, like a set_state or policy_args
        self.model_policy.observed_configs = self.observed_configs
        # fit any model/surrogates
        init_phase = self.is_init_phase()
        if not init_phase:
            self._fit_models()

    @classmethod
    def _get_config_id_split(cls, config_id: str) -> tuple[str, str]:
        # assumes config IDs of the format `[unique config int ID]_[int rung ID]`
        ids = config_id.split("_")
        _config, _budget = ids[0], ids[1]
        return _config, _budget

    def _load_previous_observations(self, previous_results):
        def index_data_split(config_id: str, config_val):
            _config_id, _budget_id = MFEIBO._get_config_id_split(config_id)
            index = int(_config_id), int(_budget_id)
            _data = [
                config_val.config,
                self.get_loss(config_val.result),
                self.get_learning_curve(config_val.result),
            ]
            return index, _data

        if len(previous_results) > 0:
            index_row = [
                tuple(index_data_split(config_id, config_val))
                for config_id, config_val in previous_results.items()
            ]
            indices, rows = zip(*index_row)
            self.observed_configs.add_data(data=list(rows), index=list(indices))

    def _handle_pending_evaluations(self, pending_evaluations):
        for config_id, config_val in pending_evaluations.items():
            _config, _budget_level = config_id.split("_")
            index = (int(_config), int(_budget_level))

            if index not in self.observed_configs.df.index:
                # TODO: Validate this
                self.observed_configs.add_data(
                    [config_val, np.nan, [np.nan]], index=index
                )
            else:
                self.observed_configs.update_data(
                    {
                        self.observed_configs.config_col: config_val,
                        self.observed_configs.perf_col: np.nan,
                        self.observed_configs.lc_col_name: [np.nan],
                    },
                    index=index,
                )

    def _fit_models(self):
        # TODO: Once done with development catch the model update exceptions
        # and skip model based suggestions if failed (karibbov)
        self._prep_model_args(self.hp_kernels, self.graph_kernels, self.pipeline_space)
        self.model_policy.set_state(self.pipeline_space, self.surrogate_model_args)
        self.model_policy.update_model()
        self.acquisition.set_state(
            self.pipeline_space,
            self.model_policy.surrogate_model,
            self.observed_configs,
            self.step_size,
        )
        self.acquisition_sampler.set_state(
            self.pipeline_space, self.observed_configs, self.step_size
        )

    def _randomly_promote(self) -> tuple[SearchSpace, int]:
        """Samples the initial design.

        With an unbiased coin toss (p=0.5) it decides whether to sample a new
        configuration or continue a partial configuration, until initial_design_size
        configurations have been sampled.
        """
        # sampling a configuration ID from the observed ones
        _config_ids = np.unique(
            self.observed_configs.df.index.get_level_values("config_id").values
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
        config.fidelity.set_value(new_fidelity)
        return config, _config_id

    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        config_id = None
        previous_config_id = None
        if self.is_init_phase(budget_based=False):
            # sample a new config till initial design size is satisfied
            self.logger.info("sampling...")
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
            assert config.fidelity is not None
            config.fidelity.set_value(self.min_budget)

            _config_id = self.observed_configs.next_config_id()
        elif self.is_init_phase(budget_based=True) or self._model_update_failed:
            # promote a config randomly if initial design size is satisfied but the
            # initial design budget has not been exhausted
            self.logger.info("promoting...")
            config, _config_id = self._randomly_promote()
        else:
            if self.count == 0:
                self.logger.info("\nPartial learning curves as initial design:\n")
                self.logger.info(f"{self.observed_configs.get_learning_curves()}\n")
            self.count += 1
            # main acquisition call here after initial design is turned off
            self.logger.info("acquiring...")
            # generates candidate samples for acquisition calculation
            assert self.pipeline_space.fidelity is not None
            samples = self.acquisition_sampler.sample(
                set_new_sample_fidelity=self.pipeline_space.fidelity.lower
            )  # fidelity values here should be the observations or min. fidelity
            # calculating acquisition function values for the candidate samples
            acq, _samples = self.acquisition.eval(  # type: ignore[attr-defined]
                x=samples, asscalar=True
            )
            # maximizing acquisition function
            _idx = np.argsort(acq)[-1]
            # extracting the config ID for the selected maximizer
            _config_id = samples.index[_samples.index.values[_idx]]
            # `_samples` should have new configs with fidelities set to as required
            # NOTE: len(samples) need not be equal to len(_samples) as `samples` contain
            # all (partials + new) configurations obtained from the sampler, but
            # in `_samples`, configs are removed that have reached maximum epochs allowed
            # NOTE: `samples` and `_samples` should share the same index values, hence,
            # avoid using `.iloc` and work with `.loc` on pandas DataFrame/Series

            # Is this "config = _samples.loc[_config_id]"?
            config = samples.loc[_config_id]
            config.fidelity.set_value(_samples.loc[_config_id].fidelity.value)
        # generating correct IDs
        if _config_id in self.observed_configs.seen_config_ids:
            config_id = f"{_config_id}_{self.get_budget_level(config)}"
            previous_config_id = f"{_config_id}_{self.get_budget_level(config) - 1}"
        else:
            config_id = f"{self.observed_configs.next_config_id()}_{self.get_budget_level(config)}"

        return config.hp_values(), config_id, previous_config_id
