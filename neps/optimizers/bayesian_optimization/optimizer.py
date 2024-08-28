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
from neps.sampling import Prior, Sampler
from neps.search_spaces.encoding import TensorEncoder, TensorPack
from neps.search_spaces.hyperparameters.categorical import CategoricalParameter

if TYPE_CHECKING:
    from botorch.models.model import Model

    from neps.search_spaces import (
        SearchSpace,
    )
    from neps.search_spaces.domain import Domain
    from neps.search_spaces.hyperparameters.float import FloatParameter
    from neps.search_spaces.hyperparameters.integer import IntegerParameter
    from neps.state import BudgetInfo, Trial


def _pibo_acq_beta_and_n(
    n_sampled_already: int,
    ndims: int,
    budget_info: BudgetInfo,
) -> tuple[float, float]:
    if budget_info.max_evaluations is not None:
        # From the PIBO paper (Section 4.1)
        # https://arxiv.org/pdf/2204.11051
        beta = budget_info.max_evaluations / 10
        return n_sampled_already, beta

    if budget_info.max_cost_budget is not None:
        # This might not work well if cost number is high
        # early on, but it will start to normalize.
        n = budget_info.used_cost_budget
        beta = budget_info.max_cost_budget / 10
        return n, beta

    # Otherwise, just some random heuristic based on the number
    # of trials and dimensionality of the search space
    # TODO: Think about and evaluate this more.
    beta = ndims**2 / 10
    return n_sampled_already, beta


# TODO: This needs to be moved to the search space class, however
# to not break the current prior based APIs used elsewhere, we can
# just manually create this here.
# We use confidence here where `0` means no confidence and `1` means
# absolute confidence. This gets translated in to std's and weights
# accordingly in a `CenteredPrior`
def _make_prior(
    parameters: dict[str, CategoricalParameter | FloatParameter | IntegerParameter],
) -> Prior:
    _mapping = {"low": 0.25, "medium": 0.5, "high": 0.75}

    domains: dict[str, Domain] = {}
    centers: dict[str, tuple[Any, float]] = {}
    categoricals: set[str] = set()
    for name, hp in parameters.items():
        domains[name] = hp.domain  # type: ignore

        if isinstance(hp, CategoricalParameter):
            categoricals.add(name)

        if hp.default is None:
            continue

        confidence_str = hp.default_confidence_choice
        confidence_score = _mapping[confidence_str]
        center = hp._default_index if isinstance(hp, CategoricalParameter) else hp.default

        centers[name] = (center, confidence_score)

    # Uses truncnorms for numerical and weighted choices categoricals
    return Prior.make_centered(
        domains=domains,
        centers=centers,
        categoricals=categoricals,
    )


class BayesianOptimization(BaseOptimizer):
    """Implements the basic BO loop."""

    def __init__(  # noqa: D417
        self,
        pipeline_space: SearchSpace,
        *,
        initial_design_size: int | None = None,
        surrogate_model: (
            Literal["gp"] | Callable[[TensorPack, torch.Tensor], Model]
        ) = "gp",
        use_priors: bool = False,
        sample_default_first: bool = False,
        device: torch.device | None = None,
        encoder: TensorEncoder | None = None,
        treat_fidelity_as_hyperparameters: bool = False,
        **kwargs: Any,  # TODO: Remove
    ):
        """Initialise the BO loop.

        Args:
            pipeline_space: Space in which to search
            initial_design_size: Number of samples used before using the surrogate model.
                If None, it will take `int(log(N) ** 2)` samples where `N` is the number
                of parameters in the search space.
            surrogate_model: Surrogate model, either a known model str or a callable
                that takes in the training data and returns a model fitted to (X, y).
            use_priors: Whether to use priors set on the hyperparameters during search.
            sample_default_first: Whether to sample the default configuration first.
            device: Device to use for the optimization.
            encoder: Encoder to use for encoding the configurations. If None, it will
                will use the default encoder.
            treat_fidelity_as_hyperparameters: Whether to treat fidelities as
                hyperparameters. If left as False, fidelities will be ignored
                and configurations will always be sampled at the maximum fidelity.

        Raises:
            ValueError: if initial_design_size < 1
            ValueError: if no kernel is provided
        """
        if any(pipeline_space.graphs):
            raise NotImplementedError("Only supports flat search spaces for now!")
        super().__init__(pipeline_space=pipeline_space)

        if initial_design_size is None:
            N = len(pipeline_space.hyperparameters)
            initial_design_size = int(max(2, math.log(N) ** 2))
        elif initial_design_size < 1:
            raise ValueError("Initial_design_size to be at least 1")

        params: dict[str, CategoricalParameter | FloatParameter | IntegerParameter] = {
            **pipeline_space.numerical,
            **pipeline_space.categoricals,
        }
        if treat_fidelity_as_hyperparameters:
            params.update(pipeline_space.fidelities)

        self.encoder = TensorEncoder.default(params) if encoder is None else encoder
        self.prior = _make_prior(params) if use_priors is True else None
        self.device = device
        self.sample_default_first = sample_default_first
        self.n_initial_design = initial_design_size
        self._get_fitted_model = (
            default_single_obj_gp if surrogate_model == "gp" else surrogate_model
        )

        self.initial_design_: list[dict[str, Any]] | None = None

    def ask(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo,
        optimizer_state: dict[str, Any],
        seed: int | None = None,
    ) -> tuple[SampledConfig, dict[str, Any]]:
        if seed is not None:
            raise NotImplementedError(
                "Seed is not yet implemented for BayesianOptimization"
            )

        space = self.pipeline_space
        config_id = str(len(trials) + 1)

        # Fill intitial design data if we don't have any...
        if self.initial_design_ is None:
            self.initial_design_ = []

            if self.sample_default_first:
                config = space.sample_default_configuration()
                self.initial_design_.append(config.hp_values())

            sampler = (
                self.prior if self.prior else Sampler.sobol(self.encoder.ncols, seed=seed)
            )
            n_samples = self.n_initial_design - len(self.initial_design_)

            x = sampler.sample(
                n_samples * 2,
                to=self.encoder.domains,
                seed=seed,
                device=self.device,
            )
            uniq_x = torch.unique(x, dim=0)
            configs = self.encoder.unpack(uniq_x[:n_samples])
            self.initial_design_.extend(configs)

        # If we havn't passed the intial design phase
        if len(trials) <= len(self.initial_design_):
            config = self.initial_design_[len(trials) - 1]
            sample = SampledConfig(id=config_id, config=config, previous_config_id=None)
            return sample, optimizer_state

        # Now we actually do the BO loop, start by encoding the data
        # TODO: Lift this into runtime, let the optimizer advertise the encoding wants...
        x_configs: list[dict[str, Any]] = []
        ys: list[float] = []
        pending: list[dict[str, Any]] = []
        for trial in trials.values():
            if trial.state.pending():
                pending.append(trial.config)
            else:
                assert trial.report is not None
                assert trial.report.loss is not None
                x_configs.append(trial.config)
                ys.append(trial.report.loss)

        x = self.encoder.pack(x_configs, device=self.device)
        x_pending = (
            None if len(pending) == 0 else self.encoder.pack(pending, device=self.device)
        )
        y = torch.tensor(ys, dtype=torch.float64, device=self.device)
        if y.ndim == 1:
            y = y.unsqueeze(1)

        model = self._get_fitted_model(x, y)

        acq = qLogExpectedImprovement(
            model,
            best_f=y.min(),
            X_pending=None if x_pending is None else x_pending.tensor,
            # Unfortunatly, there's no option to indicate that we minimize
            # the AcqFunction so we need to do some kind of transformation.
            # https://github.com/pytorch/botorch/issues/2316#issuecomment-2085964607
            objective=LinearMCObjective(weights=torch.tensor([-1.0])),
        )
        if self.prior:
            n, beta = _pibo_acq_beta_and_n(len(trials), self.encoder.ncols, budget_info)
            acq = PiboAcquisition(acq, prior=self.prior, n=n, beta=beta)

        candidates, _eis = optimize_acq(acq_fn=acq, encoder=self.encoder, acq_options={})

        assert len(candidates) == 1, "Expected only one candidate!"
        config = self.encoder.unpack(candidates)[0]

        sample = SampledConfig(id=config_id, config=config, previous_config_id=None)
        return sample, optimizer_state
