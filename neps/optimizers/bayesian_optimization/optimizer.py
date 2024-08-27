from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import torch
from botorch.acquisition import (
    LinearMCObjective,
    qLogExpectedImprovement,
)

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.optimizers.bayesian_optimization.acquisition_functions.prior_weighted import (
    PiboAcquisition,
)
from neps.optimizers.bayesian_optimization.models.gp import (
    default_single_obj_gp,
    optimize_acq,
)
from neps.optimizers.initial_design import Sobol
from neps.search_spaces.encoding import DataEncoder

if TYPE_CHECKING:
    from botorch.models.model import Model

    from neps.search_spaces import (
        SearchSpace,
    )
    from neps.search_spaces.encoding import DataPack
    from neps.state import BudgetInfo, Trial


class BayesianOptimization(BaseOptimizer):
    """Implements the basic BO loop."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        *,
        initial_design_size: int | None = None,
        surrogate_model: Literal["gp"] | Callable[[DataPack, torch.Tensor], Model] = "gp",
        use_priors: bool = False,
        sample_default_first: bool = False,
        device: torch.device | None = None,
        **kwargs: Any,  # TODO: Remove
    ):
        """Initialise the BO loop.

        Args:
            pipeline_space: Space in which to search
            initial_design_size: Number of samples used before using the surrogate model.
                If None, it will take `int(log(N) ** 2)` samples where `N` is the number
                of parameters in the search space.
            surrogate_model: Surrogate model
            use_priors: Whether to use priors set on the hyperparameters during search.

        Raises:
            ValueError: if initial_design_size < 1
            ValueError: if no kernel is provided
        """
        if initial_design_size is None:
            N = len(pipeline_space.hyperparameters)
            initial_design_size = int(max(1, math.log(N) ** 2))
        elif initial_design_size < 1:
            raise ValueError(
                "BayesianOptimization needs initial_design_size to be at least 1"
            )

        super().__init__(pipeline_space=pipeline_space)

        self.use_priors = use_priors

        # TODO: This needs to be moved to the search space class, however to not break
        # the current prior based APIs, we will create this manually here
        if use_priors:
            self._prior_confidences = {}

        self.device = device
        self.sample_default_first = sample_default_first
        self.n_initial_design = initial_design_size

        if surrogate_model == "gp":
            self._get_fitted_model = default_single_obj_gp
        else:
            self._get_fitted_model = surrogate_model

        self.encoder_: DataEncoder | None = None
        self.initial_design_: list[dict[str, Any]] | None = None

    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo,
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

        # We only do single objective for now but may as well include this
        # for when we have MO
        if y.ndim == 1:
            y = y.unsqueeze(1)

        pending = [t.config for t in trials.values() if t.state.pending()]
        if self.encoder_ is None:
            self.encoder_ = DataEncoder.default_encoder(
                self.pipeline_space,
                include_fidelities=False,
            )

        space = self.pipeline_space

        if self.initial_design_ is None:
            size = self.n_initial_design
            self.initial_design_ = []

            if self.sample_default_first:
                config = space.sample_default_configuration()
                self.initial_design_.append(config.hp_values())

            assert self.encoder_.tensors is not None
            sobol = Sobol(seed=0, encoder=self.encoder_, allow_undersampling=True)
            sobol_configs = sobol.sample(size - len(self.initial_design_))
            self.initial_design_.extend(sobol_configs)
        else:
            self.initial_design_ = []

        config_id = str(len(trials) + 1)
        if len(trials) < len(self.initial_design_):
            config = self.initial_design_[len(trials)]
            return (
                SampledConfig(id=config_id, config=config, previous_config_id=None),
                optimizer_state,
            )

        assert self.encoder_ is not None
        x = self.encoder_.encode(x_configs, device=self.device)
        if any(pending):
            x_pending = self.encoder_.encode(pending, device=self.device)
            x_pending = x_pending.tensor
            assert x_pending is not None
        else:
            x_pending = None

        model = self._get_fitted_model(x, y)

        acq = qLogExpectedImprovement(
            model,
            best_f=y.min(),
            X_pending=x_pending,
            objective=LinearMCObjective(weights=torch.tensor([-1.0])),
        )

        if self.use_priors:
            # From the PIBO paper (Section 4.1)
            # https://arxiv.org/pdf/2204.11051
            if budget_info.max_evaluations is not None:
                beta = budget_info.max_evaluations / 10
                n = budget_info.used_evaluations
            elif budget_info.max_cost_budget is not None:
                # This might not work well if cost number is high
                # early on, but it will start to normalize.
                beta = budget_info.max_cost_budget / 10
                n = budget_info.used_cost_budget

            acq = PiboAcquisition(acq, n=n, beta=beta)

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
            encoder=self.encoder_,
            acq_options={},  # options to underlying optim function of botorch
        )
        config = self.encoder_.decode_dicts(candidates)[0]
        return (
            SampledConfig(id=config_id, config=config, previous_config_id=None),
            optimizer_state,
        )
