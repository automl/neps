from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import torch
from botorch.acquisition import (
    LinearMCObjective,
    qLogExpectedImprovement,
)

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.bayesian_optimization.acquisition_functions import (
    DecayingPriorWeightedAcquisition,
)
from neps.optimizers.bayesian_optimization.models.gp import (
    default_single_obj_gp,
    optimize_acq,
)
from neps.search_spaces import (
    CategoricalParameter,
    ConstantParameter,
    FloatParameter,
    IntegerParameter,
    SearchSpace,
)
from neps.search_spaces.domain import UNIT_FLOAT_DOMAIN
from neps.search_spaces.encoding import DataEncoder

if TYPE_CHECKING:
    from botorch.models.model import Model

    from neps.search_spaces.encoding import DataPack
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
        surrogate_model: Literal["gp"] | Callable[[DataPack, torch.Tensor], Model] = "gp",
        log_prior_weighted: bool = False,
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
        device: torch.device | None = None,
        **kwargs: Any,  # TODO: Remove
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
        self.device = device
        self.sample_default_first = sample_default_first
        self.encoder: DataEncoder | None = None

        if surrogate_model == "gp":
            self._get_fitted_model = default_single_obj_gp
        else:
            self._get_fitted_model = surrogate_model

        if self.pipeline_space.has_prior:
            self.acquisition = DecayingPriorWeightedAcquisition(
                self.acquisition, log=log_prior_weighted
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

        self._cached_sobol_configs: list[dict[str, Any]] | None = None

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
        x_configs = [t.config for t in completed]
        y: torch.Tensor = torch.as_tensor(
            [t.report.loss for t in completed],
            dtype=torch.float64,
        )  # type: ignore

        # We only do single objective for now but may as well include this for when we have MO
        if y.ndim == 1:
            y = y.unsqueeze(1)

        pending = [t.config for t in trials.values() if t.state.pending()]
        if self.encoder is None:
            self.encoder = DataEncoder.default_encoder(
                self.pipeline_space,
                include_fidelities=False,
            )

        space = self.pipeline_space

        if len(trials) == 0 and self.sample_default_first and space.has_prior:
            config = space.sample_default_configuration(
                patience=self.patience, ignore_fidelity=False
            ).hp_values()

        elif len(trials) <= self._initial_design_size:
            if self._cached_sobol_configs is None:
                assert self.encoder.tensors is not None
                ndim = len(self.encoder.tensors.transformers)
                sobol = torch.quasirandom.SobolEngine(
                    dimension=ndim,
                    scramble=True,
                    seed=5,
                )

                # TODO: Need a better encapsulation of this
                x = sobol.draw(self._initial_design_size * ndim, dtype=torch.float64)
                hp_normalized_values = []
                for i, (_k, v) in enumerate(self.encoder.tensors.transformers.items()):
                    tensor = v.domain.cast(x[:, i], frm=UNIT_FLOAT_DOMAIN)
                    tensor = tensor.unsqueeze(1) if tensor.ndim == 1 else tensor
                    hp_normalized_values.append(tensor)

                tensor = torch.cat(hp_normalized_values, dim=1)
                uniq = torch.unique(tensor, dim=0)
                self._cached_sobol_configs = self.encoder.tensors.decode_dicts(uniq)

            if len(trials) <= len(self._cached_sobol_configs):
                config = self._cached_sobol_configs[len(trials) - 1]
            else:
                # The case where sobol sampling couldn't generate enough unique configs
                config = space.sample(
                    patience=self.patience, ignore_fidelity=False, user_priors=False
                ).hp_values()

        elif random.random() < self._random_interleave_prob:
            config = space.sample(
                patience=self.patience, user_priors=False, ignore_fidelity=False
            ).hp_values()
        else:
            assert self.encoder is not None
            x = self.encoder.encode(x_configs, device=self.device)
            if any(pending):
                x_pending = self.encoder.encode(pending, device=self.device)
                x_pending = x_pending.tensor
                assert x_pending is not None
            else:
                x_pending = None

            model = self._get_fitted_model(x, y)

            N_CANDIDATES_REQUIRED = 1
            N_INITIAL_RANDOM_SAMPLES = 512
            N_RESTARTS = 20

            candidates, _eis = optimize_acq(
                # TODO: We should evaluate whether LogNoisyEI is better than LogEI
                acq_fn=qLogExpectedImprovement(
                    model,
                    best_f=y.min(),
                    X_pending=x_pending,
                    # Unfortunatly, there's no option to indicate that we minimize
                    # the AcqFunction so we need to do some kind of transformation.
                    # https://github.com/pytorch/botorch/issues/2316#issuecomment-2085964607
                    objective=LinearMCObjective(weights=torch.tensor([-1.0])),
                ),
                encoder=self.encoder,
                q=N_CANDIDATES_REQUIRED,
                raw_samples=N_INITIAL_RANDOM_SAMPLES,
                num_restarts=N_RESTARTS,
                acq_options={},  # options to underlying optim function of botorch
            )
            config = self.encoder.decode_dicts(candidates)[0]

        config_id = str(len(trials) + 1)
        return SampledConfig(
            id=config_id,
            config=config,
            previous_config_id=None,
        ), optimizer_state
