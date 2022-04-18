from __future__ import annotations

import random
from typing import Any

from metahyper.api import ConfigResult, instance_from_map

from ...search_spaces import (
    CategoricalParameter,
    FloatParameter,
    GraphGrammar,
    IntegerParameter,
)
from ...search_spaces.search_space import SearchSpace
from ...utils.common import has_instance
from ...utils.result_utils import get_loss
from ..base_optimizer import BaseOptimizer
from .acquisition_functions import AcquisitionMapping
from .acquisition_functions.prior_weighted import DecayingPriorWeightedAcquisition
from .acquisition_samplers import AcquisitionSamplerMapping
from .kernels import GraphKernelMapping, StationaryKernelMapping
from .models import SurrogateModelMapping


class BayesianOptimization(BaseOptimizer):
    """Implements the basic BO loop."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        initial_design_size: int = 10,
        surrogate_model: str | Any = "gp",
        surrogate_model_fit_args: dict = None,
        optimal_assignment: bool = False,
        domain_se_kernel: str = None,
        graph_kernels: list = None,
        hp_kernels: list = None,
        acquisition: str | Any = "EI",
        log_prior_weighted: bool = False,
        acquisition_sampler: str | Any = "mutation",
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
            surrogate_model_fit_args: Arguments that will be given to the surrogate model
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
        self._num_train_x: int
        self._pending_evaluations: list = []
        self._model_update_failed = False

        if not graph_kernels:
            graph_kernels = []
            if has_instance(self.pipeline_space.values(), GraphGrammar):
                graph_kernels.append("wl")

        if not hp_kernels:
            hp_kernels = []
            if has_instance(
                self.pipeline_space.values(), FloatParameter, IntegerParameter
            ):
                hp_kernels.append("m52")
            if has_instance(self.pipeline_space.values(), CategoricalParameter):
                hp_kernels.append("hm")

        graph_kernels = [
            instance_from_map(GraphKernelMapping, kernel, "kernel", as_class=True)(
                oa=optimal_assignment,
                se_kernel=instance_from_map(
                    StationaryKernelMapping, domain_se_kernel, "se kernel"
                ),
            )
            for kernel in graph_kernels
        ]
        hp_kernels = [
            instance_from_map(StationaryKernelMapping, kernel, "kernel")
            for kernel in hp_kernels
        ]

        if not graph_kernels and not hp_kernels:
            raise ValueError("No kernels are provided!")

        self.surrogate_model = instance_from_map(
            SurrogateModelMapping,
            surrogate_model,
            name="surrogate model",
            kwargs={
                "graph_kernels": graph_kernels,
                "hp_kernels": hp_kernels,
                "vectorial_features": self.pipeline_space.get_vectorial_dim(),
                "surrogate_model_fit_args": surrogate_model_fit_args or {},
            },
        )
        self.acquisition = instance_from_map(
            AcquisitionMapping,
            acquisition,
            name="acquisition function",
        )
        # TODO: Do we want to apply this everytime?
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
            config = self.pipeline_space.sample(patience=self.patience, user_priors=True)
        elif random.random() < self._random_interleave_prob:
            config = self.pipeline_space.sample(patience=self.patience)
        elif self._num_train_x < self._initial_design_size or self._model_update_failed:
            config = self.pipeline_space.sample(patience=self.patience, user_priors=True)
        else:
            for _ in range(self.patience):
                config = self.acquisition_sampler.sample(self.acquisition)
                if config not in self._pending_evaluations:
                    break
            else:
                config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True
                )

        config_id = str(self._num_train_x + len(self._pending_evaluations) + 1)
        return config, config_id, None


# TODO: Update according to the changes above
# TODO(neps.api): this BO class gets used when
# pipeline_space.has_fidelity() == True and BO is chosen
# also when random_search is chosen, but then use no model
class BayesianOptimizationMultiFidelity(BayesianOptimization):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        initial_design_size: int = 10,
        surrogate_model_fit_args: dict = None,
        optimal_assignment: bool = False,
        domain_se_kernel: str = None,
        graph_kernels: list = None,
        hp_kernels: list = None,
        acquisition: str = "EI",
        random_interleave_prob: float = 0.0,
        patience: int = 50,
        # TODO: add eta parameter
        # TODO: update signature according to BayesianOptimization
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            initial_design_size=initial_design_size,
            surrogate_model_fit_args=surrogate_model_fit_args,
            optimal_assignment=optimal_assignment,
            domain_se_kernel=domain_se_kernel,
            graph_kernels=graph_kernels,
            hp_kernels=hp_kernels,
            acquisition=acquisition,
            random_interleave_prob=random_interleave_prob,
            patience=patience,
        )

        # TODO: set up rungs using eta and pipeline_space.fidelity

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        # TODO: Read in rungs using the config id (alternatively, use get/load state)
        super().load_results(previous_results, pending_evaluations)

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        # 1. Check if any rung has enough configs to advance the best one to the next rung

        # 2. If yes, sample this config and make sure to set previous_config_id correctly
        # to the config that is continued. This is in case the budget is something like
        # epochs and in case the user wants to load a checkpoint from the previous config
        # dir.
        previous_config_id = "TODO"

        # 3. else: sample new config on lowest rung and make sure that the acquisition
        # function is optimized always on the max fidelity only and set
        # previous_config_id = None

        config = "TODO"
        config_id = "TODO"  # Needs to take budget level into account now

        return config, config_id, previous_config_id  # type: ignore
