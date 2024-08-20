from __future__ import annotations

import random
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, Mapping

import torch

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.bayesian_optimization.acquisition_functions import (
    AcquisitionMapping,
    DecayingPriorWeightedAcquisition,
)
from neps.optimizers.bayesian_optimization.acquisition_samplers import (
    AcquisitionSamplerMapping,
)
from neps.optimizers.bayesian_optimization.models import SurrogateModelMapping
from neps.optimizers.bayesian_optimization.models.gp import ComprehensiveGP
from neps.search_spaces import (
    CategoricalParameter,
    ConstantParameter,
    FloatParameter,
    IntegerParameter,
    SearchSpace,
)
from neps.search_spaces.encoding import Encoder
from neps.utils.common import instance_from_map

if TYPE_CHECKING:
    from neps.optimizers.bayesian_optimization.acquisition_functions.base_acquisition import (
        BaseAcquisition,
    )
    from neps.optimizers.bayesian_optimization.acquisition_samplers.base_acq_sampler import (
        AcquisitionSampler,
    )
    from neps.state import BudgetInfo, Trial

# TODO(eddiebergman): Why not just include in the definition of the parameters.
CUSTOM_FLOAT_CONFIDENCE_SCORES = dict(FloatParameter.DEFAULT_CONFIDENCE_SCORES)
CUSTOM_FLOAT_CONFIDENCE_SCORES.update({"ultra": 0.05})

CUSTOM_CATEGORICAL_CONFIDENCE_SCORES = dict(
    CategoricalParameter.DEFAULT_CONFIDENCE_SCORES
)
CUSTOM_CATEGORICAL_CONFIDENCE_SCORES.update({"ultra": 8})


class BayesianOptimization(BaseOptimizer):
    """Implements the basic BO loop."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        *,
        initial_design_size: int = 10,
        surrogate_model: str | Any = "gp",
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
        prior_confidence: Literal["low", "medium", "high"] | None = None,
        sample_default_first: bool = False,
    ):
        """Initialise the BO loop.

        Args:
            pipeline_space: Space in which to search
            initial_design_size: Number of 'x' samples that need to be evaluated before
                selecting a sample using a strategy instead of randomly.
            surrogate_model: Surrogate model
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
        self._num_error_evaluations: int = 0
        self.sample_default_first = sample_default_first

        if isinstance(surrogate_model, str):
            if surrogate_model == "gp":
                self.surrogate_model = ComprehensiveGP.get_default(
                    space=pipeline_space,
                    include_fidelities=False,
                )
                self._encoder = Encoder.default(self.pipeline_space)
            else:
                raise NotImplementedError(
                    "Only 'gp' is supported as a surrogate model for now."
                )
                self.surrogate_model = instance_from_map(
                    SurrogateModelMapping,
                    surrogate_model,
                    name="surrogate model",
                    kwargs=surrogate_model_args,
                )
        else:
            self.surrogate_model = surrogate_model

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
        if self.pipeline_space.has_prior:
            for k, v in self.pipeline_space.items():
                if v.is_fidelity or isinstance(v, ConstantParameter):
                    continue
                elif isinstance(v, (FloatParameter, IntegerParameter)):
                    confidence = CUSTOM_FLOAT_CONFIDENCE_SCORES[self.prior_confidence]
                    self.pipeline_space[k].default_confidence_score = confidence
                elif isinstance(v, CategoricalParameter):
                    confidence = CUSTOM_CATEGORICAL_CONFIDENCE_SCORES[
                        self.prior_confidence
                    ]
                    self.pipeline_space[k].default_confidence_score = confidence

    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        optimizer_state: dict[str, Any],
    ) -> tuple[SampledConfig, dict[str, Any]]:
        # TODO: Lift this into runtime, let the
        # optimizer advertise the encoding wants...
        completed = [
            t
            for t in trials.values()
            if t.report is not None and t.report.loss is not None
        ]
        train_x = [t.config for t in completed]
        train_y: torch.Tensor = torch.as_tensor([t.report.loss for t in completed])  # type: ignore

        pending = [t.config for t in trials.values() if t.state.pending()]

        space = self.pipeline_space

        # TODO: This would be better if we could serialize these
        # in their encoded form. later...
        for name, hp in space.categoricals.items():
            for config in chain(train_x, pending):
                config[name] = hp.choices.index(config[name])
        for name, hp in space.graphs.items():
            for config in chain(train_x, pending):
                config[name] = hp.clone().load_from(config[name])

        if len(trials) == 0 and self.sample_default_first and space.has_prior:
            config = space.sample_default_configuration(
                patience=self.patience, ignore_fidelity=False
            )
        elif len(trials) <= self._initial_design_size:
            config = space.sample(
                patience=self.patience, user_priors=True, ignore_fidelity=False
            )
        elif random.random() < self._random_interleave_prob:
            config = space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=False
            )
        else:
            try:
                if len(pending) > 0:
                    # We want to use hallucinated results for the evaluations that have
                    # not finished yet. For this we fit a model on the finished
                    # evaluations and add these to the other results to fit another model.
                    self.surrogate_model.fit(train_x, train_y)
                    ys, _ = self.surrogate_model.predict(pending)
                    train_x += pending
                    train_y += list(ys.detach().numpy())

                # TODO: When using a GP, if we've already fit the
                # model due to the if stamet above, we only
                # need to update the model with the new points.
                # fit on all the data again, only the new points...
                self.surrogate_model.fit(train_x, train_y)
                self.acquisition.set_state(self.surrogate_model)
                self.acquisition_sampler.set_state(x=train_x, y=train_y)
                for _ in range(self.patience):
                    config = self.acquisition_sampler.sample(self.acquisition)
                    if config not in pending:
                        break
                else:
                    config = space.sample(
                        patience=self.patience, user_priors=True, ignore_fidelity=False
                    )

            except RuntimeError as e:
                self.logger.exception(
                    "Model could not be updated due to below error. Sampling will not use"
                    " the model.",
                    exc_info=e,
                )
                config = space.sample(
                    patience=self.patience, user_priors=True, ignore_fidelity=False
                )

        config_id = str(len(trials) + 1)
        return SampledConfig(
            id=config_id,
            config=config.hp_values(),
            previous_config_id=None,
        ), optimizer_state
