from __future__ import annotations

import inspect
import random

import numpy as np
import torch

from ...search_spaces import (
    CategoricalParameter,
    FloatParameter,
    GraphGrammar,
    IntegerParameter,
)
from ..base_optimizer import Optimizer
from .acquisition_function_optimization import AcquisitionOptimizerMapping
from .acquisition_function_optimization.random_sampler import RandomSampler
from .acquisition_functions import AcquisitionMapping
from .kernels import GraphKernelMapping, StationaryKernelMapping
from .models.gp import ComprehensiveGP


class BayesianOptimization(Optimizer):
    def __init__(
        self,
        pipeline_space,
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
    ):
        """Implements the basic BO loop."""

        assert 0 <= random_interleave_prob <= 1

        super().__init__()

        def _get_args_and_defaults(func):
            signature = inspect.signature(func)
            return list(signature.parameters.keys()), {
                k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            }

        if acquisition_opt_strategy_args is None:
            acquisition_opt_strategy_args = {}

        if graph_kernels is None or not graph_kernels:
            graph_kernels = list()
            if any(
                isinstance(parameter, GraphGrammar)
                for parameter in pipeline_space.values()
            ):
                graph_kernels.append("wl")

        if hp_kernels is None or not hp_kernels:
            hp_kernels = list()
            if any(
                isinstance(parameter, FloatParameter)
                or isinstance(parameter, IntegerParameter)
                for parameter in pipeline_space.values()
            ):
                hp_kernels.append("m52")

            if any(
                isinstance(parameter, CategoricalParameter)
                for parameter in pipeline_space.values()
            ):
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

        self.surrogate_model = ComprehensiveGP(
            graph_kernels=graph_kernels,
            hp_kernels=hp_kernels,
            verbose=verbose,
            vectorial_features=pipeline_space.get_vectorial_dim()
            if hasattr(pipeline_space, "get_vectorial_dim")
            else None,
        )
        acquisition_function = AcquisitionMapping[acquisition](
            surrogate_model=self.surrogate_model
        )

        if acquisition_opt_strategy in AcquisitionOptimizerMapping.keys():
            acquisition_function_opt_cls = AcquisitionOptimizerMapping[
                acquisition_opt_strategy
            ]
            arg_names, _ = _get_args_and_defaults(
                acquisition_function_opt_cls.__init__  # type: ignore[misc]
            )
            if not all(k in arg_names for k in acquisition_opt_strategy_args.keys()):
                raise ValueError("Parameter mismatch")
            self.acquisition_function_opt = acquisition_function_opt_cls(
                pipeline_space,
                acquisition_function,
                **acquisition_opt_strategy_args,
            )
        else:
            raise ValueError(
                f"Acquisition optimization strategy {acquisition_opt_strategy} is not "
                f"defined!"
            )

        self.random_interleave_prob = random_interleave_prob
        self.surrogate_model_fit_args = surrogate_model_fit_args
        self.initial_design_size = initial_design_size
        self.n_candidates = n_candidates
        self.patience = patience
        self.return_opt_details = return_opt_details

        self.random_sampler = RandomSampler(self.acquisition_function_opt.search_space)

        self.train_x: list = []
        self.train_y: list | torch.Tensor = []

        self.pending_evaluations: list = []

    def _update_model(self) -> None:
        """Updates the surrogate model and the acquisition function (optimizer)."""
        if len(self.pending_evaluations) > 0:
            self.surrogate_model.reset_XY(train_x=self.train_x, train_y=self.train_y)
            if self.surrogate_model_fit_args is not None:
                self.surrogate_model.fit(**self.surrogate_model_fit_args)
            else:
                self.surrogate_model.fit()
            ys, _ = self.surrogate_model.predict(self.pending_evaluations)
            train_x = self.train_x + self.pending_evaluations
            train_y = self.train_y + list(ys.detach().numpy())
        else:
            train_x = self.train_x
            train_y = self.train_y

        self.surrogate_model.reset_XY(train_x=train_x, train_y=train_y)
        if self.surrogate_model_fit_args is not None:
            self.surrogate_model.fit(**self.surrogate_model_fit_args)
        else:
            self.surrogate_model.fit()
        self.acquisition_function_opt.reset_surrogate_model(self.surrogate_model)
        self.acquisition_function_opt.reset_XY(x=train_x, y=train_y)

    def load_results(self, previous_results: dict, pending_evaluations: dict) -> None:
        self.train_x = [el.config for el in previous_results.values()]
        self.train_y = [
            el.result["loss"] if isinstance(el.result, dict) else np.inf
            for el in previous_results.values()
        ]
        self.pending_evaluations = [el for el in pending_evaluations.values()]
        if len(self.train_x) >= self.initial_design_size:
            self._update_model()

    def get_config_and_ids(self):
        if len(self.train_x) < self.initial_design_size:
            config = self.random_sampler.sample(1)[0]
        elif random.random() < self.random_interleave_prob:
            config = self.random_sampler.sample(1)[0]
        elif len(self.pending_evaluations) > 0:
            pending_evaluation_ids = [
                pend_eval.id[0]
                if len(pend_eval.id) == 0
                else "-".join(map(str, pend_eval.id))
                for pend_eval in self.pending_evaluations
            ]
            _patience = self.patience
            while _patience > 0:
                model_sample, _, _ = self.acquisition_function_opt.sample(
                    self.n_candidates, 1
                )
                config = model_sample[0]
                config_id = (
                    config.id if len(config.id) == 0 else "-".join(map(str, config.id))
                )
                if config_id not in pending_evaluation_ids:
                    break
                _patience -= 1
            if _patience == 0:
                config = self.random_sampler.sample(1)[0]
        else:
            model_sample, _, _ = self.acquisition_function_opt.sample(
                self.n_candidates, 1
            )
            config = model_sample[0]

        config_id = str(len(self.train_x) + len(self.pending_evaluations) + 1)
        return config, config_id, None
