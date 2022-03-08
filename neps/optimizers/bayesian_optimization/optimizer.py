from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Mapping

import metahyper
import torch
from metahyper.api import ConfigResult

from ...search_spaces import (
    CategoricalParameter,
    FloatParameter,
    GraphGrammar,
    IntegerParameter,
)
from ...search_spaces.search_space import SearchSpace
from ...utils.common import (
    get_fun_args_and_defaults,
    get_rnd_state,
    has_instance,
    set_rnd_state,
)
from ...utils.result_utils import get_loss
from .acquisition_function_optimization import AcquisitionOptimizerMapping
from .acquisition_functions import AcquisitionMapping
from .acquisition_functions.prior_weighted import DecayingPriorWeightedAcquisition
from .kernels import GraphKernelMapping, StationaryKernelMapping
from .models.gp import ComprehensiveGP


class BayesianOptimization(metahyper.Sampler):
    """Implements the basic BO loop.

    Attributes:
        surrogate_model: Gaussian process model
        train_x: Inputs previously sampled on which f has been evaluated
        train_y: Output of f on the train_x inputs
        pending_evaluations: Configurations of hyperparameters for which the
            evaluation of f is not known and will be computed, or is currently
            beeing evaluated in another process.
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
        acquisition: str = "EI",
        acquisition_opt_strategy: str = "mutation",
        acquisition_opt_strategy_args: dict = None,
        n_candidates: int = 200,
        random_interleave_prob: float = 0.0,
        patience: int = 50,
        verbose: bool = False,
        return_opt_details: bool = False,
        cost_function: None | Mapping = None,  # pylint: disable=unused-argument
    ):
        """Initialise the BO loop.

        Args:
            pipeline_space: Space in which to search
            initial_design_size: Number of 'x' samples that need to be
                evaluated before selecting a sample using a strategy instead of
                randomly.
            surrogate_model_fit_args: Arguments that will be given to the
                surrogate model (the Gaussian processes model).
            optimal_assignment: whether the optimal assignment kernel should be used.
            domain_se_kernel: Stationary kernel name
            graph_kernels: Kernels for NAS
            hp_kernels: Kernels for HPO
            acquisition: Acquisition strategy
            acquisition_opt_strategy: Acquisition function fetching strategy
            acquisition_opt_strategy_args: Arguments for the acquisition strategy function
            n_candidates: Number of configurations sampled
            random_interleave_prob: Frequency at which random configurations are sampled
                instead of configurations from the acquisition strategy.
            patience: How many times we try something that fails before giving up.
            verbose: Print details on stdout
            return_opt_details: Not used for now
            cost_function: Not used for now

        Raises:
            Exception: if no kernel is provided
            ValueError: if a string is not in a mapping
        """

        assert 0 <= random_interleave_prob <= 1

        super().__init__()

        self.pipeline_space = pipeline_space
        acquisition_opt_strategy_args = acquisition_opt_strategy_args or {}

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
            GraphKernelMapping[kernel](
                oa=optimal_assignment,
                se_kernel=None
                if domain_se_kernel is None
                else StationaryKernelMapping[domain_se_kernel],
            )
            for kernel in graph_kernels
        ]
        hp_kernels = [StationaryKernelMapping[kernel]() for kernel in hp_kernels]

        if not graph_kernels and not hp_kernels:
            raise Exception("No kernels are provided!")
        if acquisition not in AcquisitionMapping:
            raise ValueError(f"Acquisition function {acquisition} is not defined!")
        if acquisition_opt_strategy not in AcquisitionOptimizerMapping:
            raise ValueError(
                f"Acquisition optimization strategy {acquisition_opt_strategy} is not "
                f"defined!"
            )

        self.surrogate_model = ComprehensiveGP(
            graph_kernels=graph_kernels,
            hp_kernels=hp_kernels,
            verbose=verbose,
            vectorial_features=self.pipeline_space.get_vectorial_dim()
            if hasattr(self.pipeline_space, "get_vectorial_dim")
            else None,
        )
        acquisition_function = AcquisitionMapping[acquisition](
            surrogate_model=self.surrogate_model
        )
        if self.pipeline_space.has_prior:
            acquisition_function = DecayingPriorWeightedAcquisition(acquisition_function)

        acquisition_function_opt_cls = AcquisitionOptimizerMapping[
            acquisition_opt_strategy
        ]
        arg_names, _ = get_fun_args_and_defaults(
            acquisition_function_opt_cls.__init__  # type: ignore[misc]
        )
        if not all(k in arg_names for k in acquisition_opt_strategy_args):
            raise ValueError("Parameter mismatch")
        self.acquisition_function_opt = acquisition_function_opt_cls(
            self.pipeline_space,
            acquisition_function,
            **acquisition_opt_strategy_args,
        )

        self.random_interleave_prob = random_interleave_prob
        self.surrogate_model_fit_args = surrogate_model_fit_args or {}
        self.initial_design_size = initial_design_size
        self.n_candidates = n_candidates
        self.patience = patience
        self.return_opt_details = return_opt_details

        self.train_x: list = []
        self.train_y: list | torch.Tensor = []

        self.pending_evaluations: list = []

    def _update_model(self) -> None:
        """Updates the surrogate model and the acquisition function (optimizer)."""
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
        self.acquisition_function_opt.acquisition_function.update(self.surrogate_model)
        self.acquisition_function_opt.reset_XY(x=train_x, y=train_y)

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        self.train_x = [el.config for el in previous_results.values()]
        self.train_y = [get_loss(el.result) for el in previous_results.values()]
        self.pending_evaluations = [el for el in pending_evaluations.values()]
        if len(self.train_x) >= self.initial_design_size:
            self._update_model()

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        if len(self.train_x) == 0:
            # TODO: if default config sample it
            config = self.pipeline_space.sample_new(
                patience=self.patience, use_user_priors=True
            )
        elif random.random() < self.random_interleave_prob:
            config = self.pipeline_space.sample_new(patience=self.patience)
        elif len(self.train_x) < self.initial_design_size:
            config = self.pipeline_space.sample_new(
                patience=self.patience, use_user_priors=True
            )
        elif len(self.pending_evaluations) > 0:
            pending_evaluation_ids = [
                pend_eval.id[0]
                if len(pend_eval.id) == 0
                else "-".join(map(str, pend_eval.id))
                for pend_eval in self.pending_evaluations
            ]
            for _ in range(self.patience):
                model_sample, _, _ = self.acquisition_function_opt.sample(
                    self.n_candidates, 1
                )
                config = model_sample[0]
                config_id = (
                    config.id if len(config.id) == 0 else "-".join(map(str, config.id))
                )
                if config_id not in pending_evaluation_ids:  # Is this still working?
                    break
            else:
                config = self.pipeline_space.sample_new(
                    patience=self.patience, use_user_priors=True
                )
        else:
            model_sample, _, _ = self.acquisition_function_opt.sample(
                self.n_candidates, 1
            )
            config = model_sample[0]

        config_id = str(len(self.train_x) + len(self.pending_evaluations) + 1)
        return config, config_id, None

    def get_state(self) -> Any:  # pylint: disable=no-self-use
        return get_rnd_state()

    def load_state(self, state: Any):  # pylint: disable=no-self-use
        set_rnd_state(state)

    def load_config(self, config_dict):
        config = deepcopy(self.pipeline_space)
        config.load_from(config_dict)
        return config


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
        acquisition_opt_strategy: str = "mutation",
        acquisition_opt_strategy_args: dict = None,
        n_candidates: int = 200,
        random_interleave_prob: float = 0.0,
        patience: int = 50,
        verbose: bool = False,
        return_opt_details: bool = False,
        # TODO: add eta parameter
    ):
        super().__init__(
            pipeline_space,
            initial_design_size,
            surrogate_model_fit_args,
            optimal_assignment,
            domain_se_kernel,
            graph_kernels,
            hp_kernels,
            acquisition,
            acquisition_opt_strategy,
            acquisition_opt_strategy_args,
            n_candidates,
            random_interleave_prob,
            patience,
            verbose,
            return_opt_details,
        )

        # TODO: set up rungs using eta and pipeline_space.fidelity

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        # TODO: Read in rungs using the config id (alternatively, use get/load state)
        super().load_results(previous_results, pending_evaluations)

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
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

        return config, config_id, previous_config_id
