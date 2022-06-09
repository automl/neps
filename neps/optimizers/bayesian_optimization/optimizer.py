from __future__ import annotations

import random
from typing import Any

import torch
from metahyper.api import ConfigResult, instance_from_map

from ...search_spaces.search_space import SearchSpace
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

    def __init__(
        self,
        pipeline_space: SearchSpace,
        initial_design_size: int = 10,
        surrogate_model: str | Any = "gp",
        surrogate_model_args: dict = None,  # TODO: remove
        kernels: list[str | Kernel] = None,
        # optimal_assignment: bool = False, # TODO: remove
        # domain_se_kernel: str = None, # TODO: remove
        graph_kernels: list = None,  # TODO: remove
        hp_kernels: list = None,  # TODO: remove
        acquisition: str | BaseAcquisition = "EI",
        log_prior_weighted: bool = True,
        acquisition_sampler: str | AcquisitionSampler = "mutation",
        random_interleave_prob: float = 0.0,
        patience: int = 100,
        budget: None | int | float = None,
        logger=None,
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
            acquisition: Acquisition strategy
            log_prior_weighted: if to use log for prior
            acquisition_sampler: Acquisition function fetching strategy
            random_interleave_prob: Frequency at which random configurations are sampled
                instead of configurations from the acquisition strategy.
            patience: How many times we try something that fails before giving up.
            budget: Maximum budget
            logger: logger object, or None to use the neps logger

        Raises:
            ValueError: if patience < 1
            ValueError: if initial_design_size < 1
            ValueError: if random_interleave_prob is not between 0.0 and 1.0
            ValueError: if no kernel is provided
        """
        super().__init__(
            pipeline_space=pipeline_space,
            patience=patience,
            logger=logger,
            budget=budget,
        )

        if initial_design_size < 1:
            raise ValueError(
                "BayesianOptimization needs initial_design_size to be at least 1"
            )
        if not 0 <= random_interleave_prob <= 1:
            raise ValueError("random_interleave_prob should be between 0.0 and 1.0")

        self._initial_design_size = initial_design_size
        self._random_interleave_prob = random_interleave_prob
        self._model_update_failed: bool = False

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

        self.acquisition = instance_from_map(
            AcquisitionMapping,
            acquisition,
            name="acquisition function",
        )
        if self.pipeline_space.has_prior:
            self.acquisition = DecayingPriorWeightedAcquisition(
                self.acquisition, log=log_prior_weighted
            )

        self.acquisition_sampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,
            name="acquisition sampler function",
            kwargs={"patience": self.patience, "pipeline_space": self.pipeline_space},
        )

        self.train_x: list[SearchSpace] = []
        self.train_losses: list[float] = []
        self.train_costs: list[float] = []

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase/model-based search."""
        if len(self._previous_results) >= self._initial_design_size:
            return False
        return True

    def _fantasize_evaluations(self, new_x):
        """Returns x, y_loss, y_cost"""
        self.surrogate_model.fit(self.train_x, self.train_losses)
        with torch.no_grad():
            ys = self.surrogate_model.predict_mean(new_x)
        return new_x, ys.detach().tolist(), []

    def _update_optimizer_training_state(self):
        """Can be overloaded to set training state, only outside of init phase"""
        self.surrogate_model.fit(self.train_x, self.train_losses)
        self.acquisition.set_state(self.surrogate_model)
        self.acquisition_sampler.set_state(x=self.train_x, y=self.train_losses)

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, SearchSpace],
    ) -> None:
        # TODO: filter out error configs as they can not be used for modeling?
        # TODO: read out cost if they exist
        super().load_results(previous_results, pending_evaluations)
        self.train_x = [el.config for el in previous_results.values()]
        self.train_losses = [get_loss(el.result) for el in previous_results.values()]
        self.train_costs = [get_cost(el.result) for el in previous_results.values()]

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
                self._update_optimizer_training_state()
            except RuntimeError as e:
                self.logger.exception(
                    "Model could not be updated due to below error. Sampling will not use"
                    " the model.\n"
                    f"Error : {e}"
                )
                self._model_update_failed = True

    def get_new_config_id(self, config):  # pylint: disable=unused-argument
        return str(len(self._previous_results) + len(self._pending_evaluations) + 1)

    def sample_configuration_from_model(
        self,
    ) -> tuple[SearchSpace, str | None, str | None]:
        """Should return (config, config_id, previous_id) with config sampled from"""
        return self.acquisition_sampler.sample(self.acquisition), None, None

    def sample_configuration_randomly(
        self, **sampler_kwargs
    ) -> tuple[SearchSpace, str | None, str | None]:
        """Should return config, config_id, previous_id"""
        return self.pipeline_space.sample(**sampler_kwargs), None, None

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        config, config_id, previous_id = None, None, None
        if len(self._previous_results) == 0 and self._initial_design_size >= 1:
            # TODO: if default config sample it
            config, config_id, previous_id = self.sample_configuration_randomly(
                patience=self.patience, user_priors=True
            )
        elif random.random() < self._random_interleave_prob:
            config, config_id, previous_id = self.sample_configuration_randomly(
                patience=self.patience
            )
        elif self.is_init_phase() or self._model_update_failed:
            # initial design space
            config, config_id, previous_id = self.sample_configuration_randomly(
                patience=self.patience, user_priors=True
            )
        else:
            for _ in range(self.patience):
                config, config_id, previous_id = self.sample_configuration_from_model()
                if config not in self._pending_evaluations.values():
                    break
            else:
                config, config_id, previous_id = self.sample_configuration_randomly(
                    patience=self.patience, user_priors=True
                )
        if config_id is None:
            config_id = self.get_new_config_id(config)
        return config.hp_values(), config_id, previous_id
