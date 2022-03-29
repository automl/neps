from __future__ import annotations

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
from ..base_optimizer import BaseOptimizer
from .acquisition_functions import AcquisitionMapping
from .acquisition_functions.prior_weighted import DecayingPriorWeightedAcquisition
from .acquisition_samplers import AcquisitionSamplerMapping
from .kernels import GraphKernelMapping, StationaryKernelMapping
from .models.gp import ComprehensiveGP


class BayesianOptimization(BaseOptimizer):
    """Implements the basic BO loop.

    Attributes:
        surrogate_model: Gaussian process model
        train_x: Inputs previously sampled on which f has been evaluated
        train_y: Output of f on the train_x inputs
        pending_evaluations: Configurations of hyperparameters for which the evaluation of
            f is not known and will be computed, or is currently beeing evaluated in
            another process.
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        initial_design_size: int = 10,
        surrogate_model_fit_args: dict = None,
        optimal_assignment: bool = False,
        domain_se_kernel: str = None,
        graph_kernels: list = None,
        hp_kernels: list = None,
        acquisition: str | Any = "EI",
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
            surrogate_model_fit_args: Arguments that will be given to the surrogate model
                (the Gaussian processes model).
            optimal_assignment: whether the optimal assignment kernel should be used.
            domain_se_kernel: Stationary kernel name
            graph_kernels: Kernels for NAS
            hp_kernels: Kernels for HPO
            acquisition: Acquisition strategy
            acquisition_sampler: Acquisition function fetching strategy
            random_interleave_prob: Frequency at which random configurations are sampled
                instead of configurations from the acquisition strategy.
            patience: How many times we try something that fails before giving up.
            budget: Maximum budget
            logger: logger object, or None to use the neps logger

        Raises:
            Exception: if no kernel is provided
            ValueError: if a string is not in a mapping
        """
        if initial_design_size < 1:
            raise ValueError(
                "BayesianOptimization needs initial_design_size to be at least 1"
            )

        super().__init__(
            pipeline_space=pipeline_space,
            initial_design_size=initial_design_size,
            random_interleave_prob=random_interleave_prob,
            patience=patience,
            logger=logger,
            budget=budget,
        )

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
            raise Exception("No kernels are provided!")

        self.surrogate_model = ComprehensiveGP(
            graph_kernels=graph_kernels,
            hp_kernels=hp_kernels,
            vectorial_features=self.pipeline_space.get_vectorial_dim(),
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
            kwargs={"patience": self.patience},
        )
        if self.pipeline_space.has_prior:
            self.acquisition = DecayingPriorWeightedAcquisition(self.acquisition)

        self.surrogate_model_fit_args = surrogate_model_fit_args or {}

    def _update_model(self) -> None:
        """Updates the surrogate model and the acquisition function (optimizer)."""
        # TODO: filter out error configs as they can not be used for model building?
        if len(self.pending_evaluations) > 0:
            self.surrogate_model.reset_XY(train_x=self.train_x, train_y=self.train_y)
            self.surrogate_model.fit(**self.surrogate_model_fit_args)
            ys, _ = self.surrogate_model.predict(self.pending_evaluations)
            train_x = self.train_x + self.pending_evaluations
            train_y = self.train_y + list(ys.detach().numpy())
        else:
            train_x = self.train_x
            train_y = self.train_y

        self.surrogate_model.reset_XY(train_x=train_x, train_y=train_y)
        self.surrogate_model.fit(**self.surrogate_model_fit_args)
        self.acquisition.fit_on_model(self.surrogate_model)
        self.acquisition_sampler.work_with(self.pipeline_space, x=train_x, y=train_y)

    def sample(self):
        return self.acquisition_sampler.sample(self.acquisition)


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
