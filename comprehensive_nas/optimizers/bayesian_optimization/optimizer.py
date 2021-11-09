import inspect
import random
from typing import Iterable, Tuple, Union

import numpy as np

from .acquisition_function_optimization import AcquisitionOptimizerMapping
from .acqusition_functions import AcquisitionMapping
from .kernels import GraphKernelMapping, StationaryKernelMapping
from .models.gp import ComprehensiveGP

try:
    import torch
except ModuleNotFoundError:
    from comprehensive_nas.utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)

from ..core.optimizer import Optimizer
from .acquisition_function_optimization.base_acq_optimizer import AcquisitionOptimizer
from .acquisition_function_optimization.random_sampler import RandomSampler

# from .acqusition_functions.base_acqusition import BaseAcquisition


class _BayesianOptimization(Optimizer):
    def __init__(
        self,
        surrogate_model,
        acquisition_function_opt: AcquisitionOptimizer,
        random_interleave_prob: float = 0.0,
        surrogate_model_fit_args: dict = None,
        initial_design_size: int = 10,
        n_candidates: int = 200,
        return_opt_details: bool = False,
    ):
        """Implements the basic BO loop.

        Args:
            surrogate_model ([type]): surrogate model, e.g., GP
            acquisition_function (BaseAcquisition): acquisiton function, e.g., EI
            acquisition_function_opt (AcquisitionOptimizer): acqusition function optimization, e.g., mutation
            random_interleave (float, optional): interleave model samples with random samples. Defaults to 1/3.
            return_opt_details (bool, optional): holds information about model decision. Defaults to True.
        """
        assert 0 <= random_interleave_prob <= 1

        super().__init__()
        self.surrogate_model = surrogate_model
        self.acqusition_function_opt = acquisition_function_opt
        self.random_interleave_prob = random_interleave_prob
        self.surrogate_model_fit_args = surrogate_model_fit_args
        self.initial_design_size = initial_design_size
        self.n_candidates = n_candidates
        self.return_opt_details = return_opt_details

        self.random_sampler = RandomSampler(acquisition_function_opt.search_space)

        self.train_x = []
        self.train_y = []

    def initialize_model(self, x_configs: Iterable, y: Union[Iterable, torch.Tensor]):
        """Initializes the surrogate model and acquisition function (optimizer).

        Args:
            x_configs (Iterable): config.
            y (Union[Iterable, torch.Tensor]): observation.
        """
        self.train_x = []
        self.train_y = []
        self.update_model(x_configs, y)

    def update_model(
        self,
        x_configs: Iterable,
        y: Iterable,
    ) -> None:
        """Updates the surrogate model and updates the acquisiton function (optimizer).

        Args:
            x_configs (Iterable): configs.
            y (Union[Iterable, torch.Tensor]): observations.
        """
        self.train_x = x_configs
        self.train_y = y

        self.surrogate_model.reset_XY(train_x=self.train_x, train_y=self.train_y)
        if self.surrogate_model_fit_args is not None:
            self.surrogate_model.fit(**self.surrogate_model_fit_args)
        else:
            self.surrogate_model.fit()
        self.acqusition_function_opt.reset_surrogate_model(self.surrogate_model)
        self.acqusition_function_opt.reset_XY(x=self.train_x, y=self.train_y)

    def propose_new_location(
        self, batch_size: int = 5, n_candidates: int = 10
    ) -> Union[Iterable, Tuple[Iterable, dict]]:
        """Proposes new locations.

        Args:
            batch_size (int, optional): number of proposals. Defaults to 5.
            n_candidates (int, optional): how many candidates to consider. Defaults to 10.

        Returns:
            Union[Iterable, Tuple[Iterable, dict]]: proposals, (model decision information metrics)
        """
        # Ask for a location proposal from the acquisition function..
        model_batch_size = np.random.binomial(
            n=batch_size, p=1 - self.random_interleave_prob
        )

        next_x = []
        if model_batch_size > 0:
            model_samples, pool, acq_vals = self.acqusition_function_opt.sample(
                n_candidates, model_batch_size
            )
            next_x.extend(model_samples)
        elif self.return_opt_details:  # need to compute acq vals
            model_samples, pool, acq_vals = self.acqusition_function_opt.sample(
                n_candidates, 1
            )
        if batch_size - model_batch_size > 0:
            random_samples = self.random_sampler.sample(batch_size - model_batch_size)
            next_x.extend(random_samples)

        if self.return_opt_details:
            train_preds = self.surrogate_model.predict(
                self.train_x + list(next_x),
            )
            train_preds = [t.detach().cpu().numpy() for t in train_preds]
            pool_preds = self.surrogate_model.predict(
                pool,
            )
            pool_preds = [p.detach().cpu().numpy() for p in pool_preds]
            opt_details = {
                "pool": pool,
                "acq_vals": acq_vals,
                "train_preds_mean": train_preds[0],
                "train_preds_cov": train_preds[1],
                "pool_preds_mean": pool_preds[0],
                "pool_preds_cov": pool_preds[1],
            }
            return next_x, opt_details
        else:
            return next_x, None

    def get_config(self):
        if len(self.train_x) < self.initial_design_size:
            return self.random_sampler.sample(1)

        if random.random() < self.random_interleave_prob:
            return self.random_sampler.sample(1)

        model_sample, _, _ = self.acqusition_function_opt.sample(self.n_candidates, 1)
        return model_sample

    def new_result(self, job):
        config = job["config"]
        loss = job["loss"]
        # TODO temporary to be back-compatible
        self.train_x.append(config)
        self.train_y.append(loss)
        self.update_model(self.train_x, self.train_y)


class BayesianOptimization:
    def __new__(
        cls,
        pipeline_space,
        initial_design_size: int = 10,
        surrogate_model: str = "GP",
        surrogate_model_fit_args: dict = None,
        optimal_assignment=False,
        domain_se_kernel=None,
        graph_kernels=None,
        hp_kernels=None,
        acquisition: str = "EI",
        acquisition_opt_strategy: str = "mutation",
        acquisition_opt_strategy_args: dict = None,
        n_candidates: int = 200,
        random_interleave_prob: float = 0.0,
        verbose: bool = False,
    ):
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
        if hp_kernels is None or not hp_kernels:
            hp_kernels = list()

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

        surrogate_model = ComprehensiveGP(
            graph_kernels=graph_kernels, hp_kernels=hp_kernels, verbose=verbose
        )
        acquisition_function = AcquisitionMapping[acquisition](
            surrogate_model=surrogate_model
        )

        if acquisition_opt_strategy in AcquisitionOptimizerMapping.keys():
            acquisition_function_opt_cls = AcquisitionOptimizerMapping[
                acquisition_opt_strategy
            ]
            arg_names, _ = _get_args_and_defaults(acquisition_function_opt_cls.__init__)
            if not all(k in arg_names for k in acquisition_opt_strategy_args.keys()):
                raise ValueError("Parameter mismatch")
            acquisition_function_opt = acquisition_function_opt_cls(
                pipeline_space,
                acquisition_function,
                **acquisition_opt_strategy_args,
            )
        else:
            raise ValueError(
                f"Acquisition optimization strategy {acquisition_opt_strategy} is not defined!"
            )

        return _BayesianOptimization(
            surrogate_model=surrogate_model,
            acquisition_function_opt=acquisition_function_opt,
            random_interleave_prob=random_interleave_prob,
            surrogate_model_fit_args=surrogate_model_fit_args,
            initial_design_size=initial_design_size,
            n_candidates=n_candidates,
            return_opt_details=False,
        )
