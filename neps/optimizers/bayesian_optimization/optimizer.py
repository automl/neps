from __future__ import annotations

import random
from typing import Any

from metahyper.api import ConfigResult, instance_from_map

from ...search_spaces.search_space import SearchSpace
from ...utils.result_utils import get_loss
from ..base_optimizer import BaseOptimizer
from .acquisition_functions import AcquisitionMapping
from .acquisition_functions.base_acquisition import BaseAcquisition
from .acquisition_functions.prior_weighted import DecayingPriorWeightedAcquisition
from .acquisition_samplers import AcquisitionSamplerMapping
from .acquisition_samplers.base_acq_sampler import AcquisitionSampler
from .kernels.get_kernels import get_kernels
from .models import SurrogateModelMapping


class BayesianOptimization(BaseOptimizer):
    """Implements the basic BO loop."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        initial_design_size: int = 10,
        surrogate_model: str | Any = "gp",
        surrogate_model_args: dict = None,
        optimal_assignment: bool = False,
        domain_se_kernel: str = None,
        graph_kernels: list = None,
        hp_kernels: list = None,
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
            optimal_assignment: whether the optimal assignment kernel should be used.
            domain_se_kernel: Stationary kernel name
            graph_kernels: Kernels for NAS
            hp_kernels: Kernels for HPO
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
        self._num_train_x: int = 0
        self._pending_evaluations: list = []
        self._model_update_failed: bool = False

        surrogate_model_args = surrogate_model_args or {}
        graph_kernels, hp_kernels = get_kernels(
            self.pipeline_space,
            domain_se_kernel,
            graph_kernels,
            hp_kernels,
            optimal_assignment,
        )
        if "graph_kernels" not in surrogate_model_args:
            surrogate_model_args["graph_kernels"] = graph_kernels
        if "hp_kernels" not in surrogate_model_args:
            surrogate_model_args["hp_kernels"] = hp_kernels

        if (
            not surrogate_model_args["graph_kernels"]
            and not surrogate_model_args["hp_kernels"]
        ):
            raise ValueError("No kernels are provided!")

        if "vectorial_features" not in surrogate_model_args:
            surrogate_model_args[
                "vectorial_features"
            ] = self.pipeline_space.get_vectorial_dim()

        self.surrogate_model = instance_from_map(
            SurrogateModelMapping,
            surrogate_model,
            name="surrogate model",
            kwargs=surrogate_model_args,
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

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        # TODO: filter out error configs as they can not be used for modeling?
        # TODO: read out cost if they exist
        train_x = [el.config for el in previous_results.values()]
        train_y = [get_loss(el.result) for el in previous_results.values()]
        self._num_train_x = len(train_x)
        self._pending_evaluations = [el for el in pending_evaluations.values()]
        # TODO: Unify initial design with MF-BO
        if self._num_train_x >= self._initial_design_size:
            try:
                if len(self._pending_evaluations) > 0:
                    # We want to use hallucinated results for the evaluations that have
                    # not finished yet. For this we fit a model on the finished
                    # evaluations and add these to the other results to fit another model.
                    self.surrogate_model.fit(train_x, train_y)
                    ys, _ = self.surrogate_model.predict(self._pending_evaluations)
                    train_x += self._pending_evaluations
                    train_y += list(ys.detach().numpy())

                self.surrogate_model.fit(train_x, train_y)
                self.acquisition.set_state(self.surrogate_model)
                self.acquisition_sampler.set_state(x=train_x, y=train_y)

                self._model_update_failed = False
            except RuntimeError:
                self.logger.exception(
                    "Model could not be updated due to below error. Sampling will not use"
                    " the model."
                )
                self._model_update_failed = True

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        if self._num_train_x == 0 and self._initial_design_size >= 1:
            # TODO: if default config sample it
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
        elif random.random() < self._random_interleave_prob:
            config = self.pipeline_space.sample(
                patience=self.patience, ignore_fidelity=False
            )
        elif self._num_train_x < self._initial_design_size or self._model_update_failed:
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
        else:
            for _ in range(self.patience):
                config = self.acquisition_sampler.sample(self.acquisition)
                if config not in self._pending_evaluations:
                    break
            else:
                config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True, ignore_fidelity=False
                )

        config_id = str(self._num_train_x + len(self._pending_evaluations) + 1)
        return config, config_id, None
