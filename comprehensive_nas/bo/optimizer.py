from typing import Tuple

from ..core.optimizer import Optimizer


class BayesianOptimization(Optimizer):
    def __init__(
        self,
        surrogate_model,
        acquisition_function,
        acquisition_function_opt=None,
    ):
        super().__init__()
        self.surrogate_model = surrogate_model
        self.acquisition_function = acquisition_function
        self.acqusition_function_opt = acquisition_function_opt

    def initialize_model(self, x_configs, y):
        self.update_model(x_configs, y)

    def update_model(self, x_configs, y):
        self.surrogate_model.reset_XY(x_configs, y)
        self.surrogate_model.fit()

        self.acquisition_function.reset_surrogate_model(self.surrogate_model)

        self.acqusition_function_opt.reset_XY(x_configs, y)

    def propose_new_location(
        self, batch_size: int = 5, pool_size: int = 10
    ) -> Tuple[Tuple, dict]:
        # create candidate pool
        pool = self.acqusition_function_opt.sample(pool_size)

        # Ask for a location proposal from the acquisition function..
        next_x, eis, _ = self.acquisition_function.propose_location(
            top_n=batch_size, candidates=pool.copy()
        )

        opt_details = {"pool": pool, "eis": eis}

        return next_x, opt_details
