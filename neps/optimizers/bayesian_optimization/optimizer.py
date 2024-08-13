from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING, Literal
from typing_extensions import override
import numpy as np

from neps.state.optimizer import BudgetInfo
from neps.utils.types import ConfigResult, RawConfig
from neps.utils.common import instance_from_map
from neps.search_spaces import SearchSpace
from neps.optimizers.base_optimizer import BaseOptimizer
from neps.optimizers.bayesian_optimization.acquisition_functions import (
    AcquisitionMapping,
    DecayingPriorWeightedAcquisition,
)
from neps.optimizers.bayesian_optimization.acquisition_samplers import (
    AcquisitionSamplerMapping,
)
from neps.optimizers.bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)
from neps.optimizers.bayesian_optimization.kernels.get_kernels import get_kernels
from neps.optimizers.bayesian_optimization.models import SurrogateModelMapping

if TYPE_CHECKING:
    from neps.optimizers.bayesian_optimization.acquisition_functions.base_acquisition import (
        BaseAcquisition,
    )


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
        log_prior_weighted: bool = False,
        acquisition_sampler: str | AcquisitionSampler = "mutation",
        random_interleave_prob: float = 0.0,
        patience: int = 100,
        budget: None | int | float = None,
        ignore_errors: bool = False,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        logger=None,
        disable_priors: bool = False,
        prior_confidence: Literal["low", "medium", "high"] = None,
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
            pipeline_space.initial_prior = {}
            self.prior_confidence = None
        else:
            self.prior_confidence = prior_confidence

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

        self._initial_design_size = initial_design_size
        self._random_interleave_prob = random_interleave_prob
        self._num_train_x: int = 0
        self._num_error_evaluations: int = 0
        self._pending_evaluations: list = []
        self._model_update_failed: bool = False
        self.sample_default_first = sample_default_first

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
            surrogate_model_args["vectorial_features"] = (
                self.pipeline_space.get_vectorial_dim()
            )

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

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase/model-based search."""
        if self._num_train_x >= self._initial_design_size:
            return False
        return True

    @override
    def load_optimization_state(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, SearchSpace],
        budget_info: BudgetInfo | None,
        optimizer_state: dict[str, Any],
    ) -> None:
        train_x = [el.config for el in previous_results.values()]
        train_y = [self.get_loss(el.result) for el in previous_results.values()]
        if self.ignore_errors:
            train_x = [x for x, y in zip(train_x, train_y) if y != "error"]
            train_y_no_error = [y for y in train_y if y != "error"]
            self._num_error_evaluations = len(train_y) - len(train_y_no_error)
            train_y = train_y_no_error
        self._num_train_x = len(train_x)
        self._pending_evaluations = [el for el in pending_evaluations.values()]
        if not self.is_init_phase():
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
            except RuntimeError as runtime_error:
                self.logger.exception(
                    "Model could not be updated due to below error. Sampling will not use"
                    " the model."
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

    def get_config_and_ids(self) -> tuple[RawConfig, str, str | None]:
        seed = np.random.default_rng()
        space = self.pipeline_space

        if self._num_train_x == 0 and self.sample_default_first and space.has_prior:
            config = space.default_configuration

        elif self._num_train_x == 0 and self._initial_design_size >= 1:
            config = space.sample(around=space.initial_prior, seed=seed)

        elif random.random() < self._random_interleave_prob:
            config = space.sample(seed=seed)

        elif self.is_init_phase() or self._model_update_failed:
            config = space.sample(around=space.initial_prior, seed=seed)

        else:
            for _ in range(self.patience):
                config = self.acquisition_sampler.sample(self.acquisition)
                if config not in self._pending_evaluations:
                    break
            else:
                config = space.sample(around=space.initial_prior, seed=seed)

        if space.fidelity is not None:
            max_fidelity = space.fidelity.domain.upper
            config = config.at_fidelity(max_fidelity)

        config_id = str(
            self._num_train_x
            + self._num_error_evaluations
            + len(self._pending_evaluations)
            + 1
        )
        return config.values, config_id, None
