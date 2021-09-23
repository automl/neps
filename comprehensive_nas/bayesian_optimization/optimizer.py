from typing import Iterable, Tuple, Union

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    from comprehensive_nas.utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)

from ..core.optimizer import Optimizer
from .acquisition_function_optimization.base_acq_optimizer import AcquisitionOptimizer
from .acquisition_function_optimization.random_sampler import RandomSampler
from .acqusition_functions.base_acqusition import BaseAcquisition


class BayesianOptimization(Optimizer):
    def __init__(
        self,
        surrogate_model,
        acquisition_function: BaseAcquisition,
        acquisition_function_opt: AcquisitionOptimizer,
        random_interleave: float = 0.0,
        return_opt_details: bool = False,
        surrogate_model_fit_args: dict = None,
    ):
        """Implements the basic BO loop.

        Args:
            surrogate_model ([type]): surrogate model, e.g., GP
            acquisition_function (BaseAcquisition): acquisiton function, e.g., EI
            acquisition_function_opt (AcquisitionOptimizer): acqusition function optimization, e.g., mutation
            random_interleave (float, optional): interleave model samples with random samples. Defaults to 1/3.
            return_opt_details (bool, optional): holds information about model decision. Defaults to True.
        """
        assert 0 <= random_interleave <= 1

        super().__init__()
        self.surrogate_model = surrogate_model
        self.acquisition_function = acquisition_function
        self.acqusition_function_opt = acquisition_function_opt
        self.random_interleave = random_interleave
        self.return_opt_details = return_opt_details
        self.surrogate_model_fit_args = surrogate_model_fit_args

        self.random_sampler = RandomSampler(acquisition_function_opt.objective)

        self.train_x = None
        self.train_y = None

    def initialize_model(self, x_configs: Iterable, y: Union[Iterable, torch.Tensor]):
        """Initializes the surrogate model and acquisition function (optimizer).

        Args:
            x_configs (Iterable): config.
            y (Union[Iterable, torch.Tensor]): observation.
        """
        self.update_model(x_configs, y)

    def update_model(
        self,
        x_configs: Iterable,
        y: Union[Iterable, torch.Tensor],
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
        self.acquisition_function.reset_surrogate_model(
            surrogate_model=self.surrogate_model
        )
        self.acqusition_function_opt.reset_XY(x=self.train_x, y=self.train_y)

    def propose_new_location(
        self, batch_size: int = 5, pool_size: int = 10
    ) -> Union[Iterable, Tuple[Iterable, dict]]:
        """Proposes new locations.

        Args:
            batch_size (int, optional): number of proposals. Defaults to 5.
            pool_size (int, optional): how many candidates to consider. Defaults to 10.

        Returns:
            Union[Iterable, Tuple[Iterable, dict]]: proposals, (model decision information metrics)
        """
        # create candidate pool
        pool = self.acqusition_function_opt.sample(pool_size)

        # Ask for a location proposal from the acquisition function..
        model_batch_size = np.random.binomial(n=batch_size, p=1 - self.random_interleave)

        next_x = []
        if model_batch_size > 0:
            model_samples, acq_vals, _ = self.acquisition_function.propose_location(
                top_n=model_batch_size, candidates=pool
            )
            next_x.extend(model_samples)
        elif self.return_opt_details:  # need to compute acq vals
            _, acq_vals, _ = self.acquisition_function.propose_location(
                top_n=1, candidates=pool
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
