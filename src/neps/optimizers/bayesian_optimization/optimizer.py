from __future__ import annotations

import random
from typing import Any

import torch
from typing_extensions import Literal

from metahyper import ConfigResult, instance_from_map

from ...search_spaces.search_space import SearchSpace
from ...utils.common import disabled
from ...utils.result_utils import get_cost, get_loss
from ..base_optimizer import BaseOptimizer
from .acquisition_functions import AcquisitionMapping
from .acquisition_functions.base_acquisition import BaseAcquisition
from .acquisition_functions.prior_weighted import DecayingPriorWeightedAcquisition
from .acquisition_samplers import AcquisitionSamplerMapping
from .acquisition_samplers.base_acq_sampler import AcquisitionSampler
from .kernels import Kernel
from .models import SurrogateModelMapping


class BayesianOptimization(BaseOptimizer):
    """Implements the basic BO loop."""

    USES_COST_MODEL = False  # Set to True in a child class to use a cost model

    def __init__(
        self,
        pipeline_space: SearchSpace,
        initial_design_size: int = 10,
        surrogate_model: str | Any = "gp",
        surrogate_model_args: dict = None,  # TODO: remove
        # optimal_assignment: bool = False, # TODO: remove
        # domain_se_kernel: str = None, # TODO: remove
        kernels: list[str | Kernel] = None,
        graph_kernels: list = None,  # TODO: remove
        hp_kernels: list = None,  # TODO: remove
        cost_model: str | Literal["same"] | Any = "same",
        cost_model_kernels: list[str | Kernel] | None | Literal["same"] = "same",
        acquisition: str | BaseAcquisition = "EI",
        log_prior_weighted: bool = True,
        acquisition_sampler: str | AcquisitionSampler | None = None,
        random_interleave_prob: float = 0.0,
        patience: int = 100,
        budget: None | int | float = None,
        ignore_errors: bool = False,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        logger=None,
        disable_priors: bool = False,
        sample_default_first: bool = False,
    ):
        """Initialise the BO loop.

        Args:
            pipeline_space: Space in which to search
            initial_design_size: Number of 'x' samples that need to be evaluated before
                selecting a sample using a strategy instead of randomly.
            surrogate_model: Surrogate model
            surrogate_model_args: Arguments that will be given to the surrogate model
                (the Gaussian processes model).
            kernels: Kernels for NAS-HPO
            hp_kernels: Kernels for HPO (deprecated, should use 'kernels')
            graph_kernels: Kernels for NAS (deprecated, should use 'kernels')
            cost_model: if USES_COST_MODEL, the surrogate model used to model the cost,
                or "same" to use the value of the surrogate_model argument.
            cost_model_kernels: kernels for the cost model, or "same" to use the value
                of the kernel argument.
            acquisition: Acquisition strategy
            log_prior_weighted: if to use log for prior
            acquisition_sampler: Acquisition function fetching strategy.
                Default is "mutation" if allowed on the search space, and "random" otherwise.
            random_interleave_prob: Frequency at which random configurations are sampled
                instead of configurations from the acquisition strategy.
            patience: How many times we try something that fails before giving up.
            budget: Maximum budget
            ignore_errors: Ignore hyperparameter settings that threw an error and do not
                raise an error. Error configs still count towards max_evaluations_total.
            loss_value_on_error: Setting this and cost_value_on_error to any float will
                supress any error during bayesian optimization and will use given loss
                value instead. default: None
            cost_value_on_error: Setting this and loss_value_on_error to any float will
                supress any error during bayesian optimization and will use given cost
                value instead. default: None
            logger: logger object, or None to use the neps logger
            disable_priors: allows to choose between BO and piBO regardless the search
                space definition
            sample_default_first: if True and a default prior exists, the first sampel is
                the default configuration

        Raises:
            ValueError: if patience < 1
            ValueError: if initial_design_size < 1
            ValueError: if random_interleave_prob is not between 0.0 and 1.0
            ValueError: if no kernel is provided
        """
        if disable_priors:
            pipeline_space.has_prior = False

        super().__init__(
            pipeline_space=pipeline_space,
            patience=patience,
            logger=logger,
            budget=budget,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
        )

        if initial_design_size < 1:
            raise ValueError(
                "BayesianOptimization needs initial_design_size to be at least 1"
            )
        if not 0 <= random_interleave_prob <= 1:
            raise ValueError("random_interleave_prob should be between 0.0 and 1.0")

        if not self.USES_COST_MODEL:
            if cost_model not in ["same", None]:
                raise ValueError("This optimizer don't use a cost model")
            cost_model = None

        self._initial_design_size = initial_design_size
        self._random_interleave_prob = random_interleave_prob
        self._num_train_x: int = 0
        self._num_error_evaluations: int = 0
        self._model_update_failed: bool = False
        self.cost_model: Any = None
        self.sample_default_first = sample_default_first

        if graph_kernels:
            kernels = (kernels or []) + graph_kernels
            logger.warn(
                "Using 'graph_kernels' is deprecated and you should directly use the 'kernels' argument"
            )

        if hp_kernels:
            kernels = (kernels or []) + hp_kernels
            logger.warn(
                "Using 'hp_kernels' is deprecated and you should directly use the 'kernels' argument"
            )

        if surrogate_model_args:
            logger.warn(
                "Using 'surrogate_model_args' is deprecated. You can use a partial for the surrogate model instead: ('model_name', {args...})"
            )

        self.surrogate_model = instance_from_map(
            SurrogateModelMapping,
            surrogate_model,
            name="surrogate model",
            kwargs={
                "pipeline_space": pipeline_space,
                "kernels": kernels,
                **(surrogate_model_args or {}),
            },
        )

        if self.USES_COST_MODEL:
            if cost_model_kernels == "same":
                cost_model_kernels = kernels
            self.cost_model = instance_from_map(
                SurrogateModelMapping,
                surrogate_model if cost_model == "same" else cost_model,
                name="cost surrogate model",
                kwargs={
                    "pipeline_space": pipeline_space,
                    "kernels": kernels,
                },
            )

        self.acquisition = instance_from_map(
            AcquisitionMapping,
            acquisition,
            name="acquisition function",
        )
        if self.pipeline_space.has_prior:
            self.acquisition = DecayingPriorWeightedAcquisition(
                self.acquisition, log=log_prior_weighted
            )

        if acquisition_sampler is None:
            if pipeline_space.mutate.__func__ is disabled:  # type: ignore
                acquisition_sampler = "random"
            else:
                acquisition_sampler = "mutation"
        self.acquisition_sampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,
            name="acquisition sampler function",
            kwargs={"patience": self.patience, "pipeline_space": self.pipeline_space},
        )

        self.train_x: list[SearchSpace] = []
        self.train_losses: list[float] = []
        self.train_costs: list[float] = []
        self.fantasized_losses: list[float] = []
        self.fantasized_costs: list[float] = []

    def _fantasize_evaluations(self, new_x):
        """Returns x, y_loss, y_cost"""
        self.surrogate_model.fit(self.train_x, self.train_losses)
        with torch.no_grad():
            predicted_losses = self.surrogate_model.predict_mean(new_x)
            predicted_losses = predicted_losses.detach().tolist()

        predicted_costs = None
        if self.USES_COST_MODEL:
            self.cost_model.fit(self.train_x, self.train_costs)
            with torch.no_grad():
                predicted_costs = self.cost_model.predict_mean(new_x)
            predicted_costs = predicted_costs.detach().tolist()

        return new_x, predicted_losses, predicted_costs

    def _update_optimizer_training_state(self):
        """Can be overloaded to set training state, called only outside of init phase."""
        if not self.is_init_phase():
            self.surrogate_model.fit(self.train_x, self.train_losses)
            if self.USES_COST_MODEL:
                self.cost_model.fit(self.train_x, self.train_costs)

            self.acquisition.set_state(self.surrogate_model, cost_model=self.cost_model)
            self.acquisition_sampler.set_state(x=self.train_x, y=self.train_losses)

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, SearchSpace],
    ) -> None:
        """Interface with the neps library to load the results.

        To redefine in an child class without changing the expected behavior,
        you can overload _fantasize_evaluations and _update_optimizer_training_state."""
        # TODO: filter out error configs as they can not be used for modeling?
        # TODO: read out cost if they exist
        super().load_results(previous_results, pending_evaluations)
        self.train_x = [el.config for el in previous_results.values()]
        self.train_losses = [get_loss(el.result) for el in previous_results.values()]
        self.train_costs = [get_cost(el.result) for el in previous_results.values()]
        self.fantasized_losses, self.fantasized_costs = [], []
        self._num_train_x = len(self.train_x)

        if self.ignore_errors:
            self.train_x = [
                x for x, y in zip(self.train_x, self.train_losses) if y != "error"
            ]
            train_losses_no_error = [y for y in self.train_losses if y != "error"]
            self._num_error_evaluations = len(self.train_losses) - len(
                train_losses_no_error
            )
            self.train_losses = train_losses_no_error
        self._num_train_x = len(self.train_x)
        self._model_update_failed = False

        if not self.is_init_phase():
            try:
                if len(self._pending_evaluations) > 0:
                    # We want to use hallucinated results for the evaluations that have
                    # not finished yet. For this we fit a model on the finished
                    # evaluations and add these to the other results to fit another model.
                    new_x, new_losses, new_costs = self._fantasize_evaluations(
                        list(self._pending_evaluations.values())
                    )
                    self.train_x.extend(new_x)
                    self.train_losses.extend(new_losses)
                    self.train_costs.extend(new_costs)
                    self.fantasized_losses, self.fantasized_costs = new_losses, new_costs
                self._update_optimizer_training_state()
            except RuntimeError as runtime_error:
                self.logger.exception(
                    "Model could not be updated due to below error. Sampling will not use"
                    " the model.\n"
                    f"Error : {runtime_error}"
                )
                if self.loss_value_on_error is None or self.cost_value_on_error is None:
                    raise ValueError(
                        "A RuntimeError happened and "
                        "loss_value_on_error or cost_value_on_error "
                        "value is not provided, please fix the error or "
                        "provide the values to continue without "
                        "updating the model"
                    ) from runtime_error
                self._model_update_failed = True

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase/model-based search."""
        if len(self._previous_results) >= self._initial_design_size:
            return False
        return True

    def sample_configuration_from_model(
        self,
    ) -> tuple[SearchSpace, str | None, str | None]:
        """Called when a configuration should be sampled using the model,
        after the initialization phase.

        Returns:
            config: the new configuration
            config_id: a unique id, or None to use the id given by get_new_config_id
            previous_id: the id of the previous configuration if this is a continuation"""
        config = self.acquisition_sampler.sample(
            self.acquisition, constraint=self.sampling_constraint
        )
        return config, None, None

    def sample_configuration_randomly(
        self,
        user_priors=True,
    ) -> tuple[SearchSpace, str | None, str | None]:
        """Called when a configuration should be sampled without using the model,
        mainly for initialization phase.

        Args:
            user_priors: if we are in a case where the user_priors should be used

        Returns:
            config: the new configuration
            config_id: a unique id, or None to use the id given by get_new_config_id
            previous_id: the id of the previous configuration if this is a continuation
        """
        return (
            self.random_sampler.sample(
                user_priors=user_priors, constraint=self.sampling_constraint
            ),
            None,
            None,
        )

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        """Interface with the neps library to sample a new configuration.

        To redefine in an child class without changing the expected behavior, you can overload
        sample_configuration_from_model, sample_configuration_randomly
        and is_init_phase."""
        # pylint: disable=unused-variable
        config, config_id, previous_id = None, None, None

        if (
            self._num_train_x == 0
            and self.sample_default_first
            and self.pipeline_space.has_prior
        ):
            config = self.pipeline_space.sample_default_configuration(
                patience=self.patience
            )
        elif len(self._previous_results) == 0 and self._initial_design_size >= 1:
            # TODO: if default config sample it
            config, config_id, previous_id = self.sample_configuration_randomly(
                user_priors=True
            )
        elif random.random() < self._random_interleave_prob:
            config, config_id, previous_id = self.sample_configuration_randomly()
        elif self.is_init_phase() or self._model_update_failed:
            # initial design space
            config, config_id, previous_id = self.sample_configuration_randomly(
                user_priors=True
            )
        else:
            for _ in range(self.patience):
                config, config_id, previous_id = self.sample_configuration_from_model()
                if config not in self._pending_evaluations.values():
                    break
            else:
                config, config_id, previous_id = self.sample_configuration_randomly(
                    user_priors=True
                )
                # TODO: use self.get_new_config_id(config) instead
        if config_id is None:
            config_id = str(
                self._num_train_x
                + self._num_error_evaluations
                + len(self._pending_evaluations)
                + 1
            )
        return config.hp_values(), config_id, None
