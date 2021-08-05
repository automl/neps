from typing import Iterable, Tuple, Union

try:
    import torch
except ModuleNotFoundError:
    from install_dev_utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)

from ..core.optimizer import Optimizer


class BayesianOptimization(Optimizer):
    def __init__(
        self,
        surrogate_model,
        acquisition_function,
        acquisition_function_opt=None,
        return_opt_details: bool = True,
    ):
        super().__init__()
        self.surrogate_model = surrogate_model
        self.acquisition_function = acquisition_function
        self.acqusition_function_opt = acquisition_function_opt
        self.return_opt_details = return_opt_details

        self.train_x = None
        self.train_y = None

    def initialize_model(self, x_configs: Iterable, y: Union[Iterable, torch.Tensor]):
        self.update_model(x_configs, y)

    def update_model(self, x_configs: Iterable, y: Union[Iterable, torch.Tensor]) -> None:
        self.train_x = x_configs
        self.train_y = y

        self.surrogate_model.reset_XY(train_x=self.train_x, train_y=self.train_y)
        self.surrogate_model.fit()
        self.acquisition_function.reset_surrogate_model(
            surrogate_model=self.surrogate_model
        )
        self.acqusition_function_opt.reset_XY(x=self.train_x, y=self.train_y)

    def propose_new_location(
        self, batch_size: int = 5, pool_size: int = 10
    ) -> Union[Iterable, Tuple[Iterable, dict]]:
        # create candidate pool
        pool = self.acqusition_function_opt.sample(pool_size)

        # Ask for a location proposal from the acquisition function..
        next_x, eis, _ = self.acquisition_function.propose_location(
            top_n=batch_size, candidates=pool
        )

        if self.return_opt_details:
            train_preds = self.surrogate_model.predict(
                self.train_x,
            )
            train_preds = [t.detach().cpu().numpy() for t in train_preds]
            pool_preds = self.surrogate_model.predict(
                pool,
            )
            pool_preds = [p.detach().cpu().numpy() for p in pool_preds]
            opt_details = {
                "pool": pool,
                "eis": eis,
                "train_preds_mean": train_preds[0],
                "train_preds_cov": train_preds[1],
                "pool_preds_mean": pool_preds[0],
                "pool_preds_cov": pool_preds[1],
            }
            return next_x, opt_details
        else:
            return next_x
