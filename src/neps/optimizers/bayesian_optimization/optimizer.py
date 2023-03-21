from __future__ import annotations

import random
from typing import Any

from typing_extensions import Literal

from metahyper import ConfigResult, instance_from_map

from ...search_spaces.hyperparameters.categorical import (
    CATEGORICAL_CONFIDENCE_SCORES,
    CategoricalParameter,
)
from ...search_spaces.hyperparameters.constant import ConstantParameter
from ...search_spaces.hyperparameters.float import FLOAT_CONFIDENCE_SCORES, FloatParameter
from ...search_spaces.hyperparameters.integer import IntegerParameter
from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer
from .acquisition_functions import AcquisitionMapping
from .acquisition_functions.base_acquisition import BaseAcquisition
from .acquisition_functions.prior_weighted import DecayingPriorWeightedAcquisition
from .acquisition_samplers import AcquisitionSamplerMapping
from .acquisition_samplers.base_acq_sampler import AcquisitionSampler
from .kernels.get_kernels import get_kernels
from .models import SurrogateModelMapping

CUSTOM_FLOAT_CONFIDENCE_SCORES = FLOAT_CONFIDENCE_SCORES.copy()
CUSTOM_FLOAT_CONFIDENCE_SCORES.update({"ultra": 0.05})

CUSTOM_CATEGORICAL_CONFIDENCE_SCORES = CATEGORICAL_CONFIDENCE_SCORES.copy()
CUSTOM_CATEGORICAL_CONFIDENCE_SCORES.update({"ultra": 8})


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
            pipeline_space.has_prior = False
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
        self._enhance_priors()
        print()

    def _enhance_priors(self, confidence_score: dict = None) -> None:
        """Only applicable when priors are given along with a confidence.

        Args:
            confidence_score: dict
                The confidence scores for the 2 major variable types.
                Example: {"categorical": 5.2, "numeric": 0.15}
        """
        if self.prior_confidence is None:
            return
        if (
            hasattr(self.pipeline_space, "has_prior")
            and not self.pipeline_space.has_prior
        ):
            return
        for k, v in self.pipeline_space.items():
            if v.is_fidelity or isinstance(v, ConstantParameter):
                continue
            elif isinstance(v, (FloatParameter, IntegerParameter)):
                if confidence_score is None:
                    confidence = CUSTOM_FLOAT_CONFIDENCE_SCORES[self.prior_confidence]
                else:
                    confidence = confidence_score["numeric"]
                self.pipeline_space[k].default_confidence_score = confidence
            elif isinstance(v, CategoricalParameter):
                if confidence_score is None:
                    confidence = CUSTOM_CATEGORICAL_CONFIDENCE_SCORES[
                        self.prior_confidence
                    ]
                else:
                    confidence = confidence_score["categorical"]
                self.pipeline_space[k].default_confidence_score = confidence
        return

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase/model-based search."""
        if self._num_train_x >= self._initial_design_size:
            return False
        return True

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
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

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        if (
            self._num_train_x == 0
            and self.sample_default_first
            and self.pipeline_space.has_prior
        ):
            config = self.pipeline_space.sample_default_configuration(
                patience=self.patience, ignore_fidelity=False
            )
        elif self._num_train_x == 0 and self._initial_design_size >= 1:
            config = self.pipeline_space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
        elif random.random() < self._random_interleave_prob:
            config = self.pipeline_space.sample(
                patience=self.patience, ignore_fidelity=False
            )
        elif self.is_init_phase() or self._model_update_failed:
            # initial design space
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

        config_id = str(
            self._num_train_x
            + self._num_error_evaluations
            + len(self._pending_evaluations)
            + 1
        )
        return config.hp_values(), config_id, None
