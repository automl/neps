# from typing import Iterable, Union
#
# import numpy as np
# import torch
#
# from .base_acquisition import BaseAcquisition

# class GaussianProcessUpperConfidenceBound(BaseAcquisition):
#     def __init__(
#         self,
#         augmented_ei: bool = False,
#         beta: float = 1.0,
#         beta_decay: float = 0.95,
#         in_fill: str = "best",
#         log_ei: bool = False,
#     ):
#         """Calculates vanilla GP-UCB over the candidate set.
#
#         Args:
#             surrogate_model: surrogate model, e.g., GP.
#             augmented_ei: Using the Augmented EI heuristic modification to the standard
#                 expected improvement algorithm according to Huang (2006).
#             xi: manual exploration-exploitation trade-off parameter.
#             in_fill: the criterion to be used for in-fill for the determination of mu_star
#                 'best' means the empirical best observation so far (but could be
#                 susceptible to noise), 'posterior' means the best *posterior GP mean*
#                 encountered so far, and is recommended for optimization of more noisy
#                 functions. Defaults to "best".
#             log_ei: log-EI if true otherwise usual EI.
#         """
#         super().__init__()
#
#         if in_fill not in ["best", "posterior"]:
#             raise ValueError(f"Invalid value for in_fill ({in_fill})")
#         self.augmented_ei = augmented_ei
#         self.beta = beta
#         self.beta_decay = beta_decay
#         self.t = 0  # optimization trace size
#         self.in_fill = in_fill
#         self.log_ei = log_ei
#         self.incumbent = None
#
#     def eval(
#         self, x: Iterable, asscalar: bool = False
#     ) -> Union[np.ndarray, torch.Tensor, float]:
#         """
#         Return GP-UCB utility score at each point
#         """
#         assert self.incumbent is not None, "EI function not fitted on model"
#         try:
#             mu, cov = self.surrogate_model.predict(x)
#         except ValueError as e:
#             raise e
#             # return -1.0  # in case of error. return ei of -1
#         std = torch.sqrt(torch.diag(cov))
#         # TODO: option for more sophisticated exploration weight parameter
#         gp_ucb = mu + torch.sqrt(self.beta) * std
#         # with each round/batch of evaluation the exploration factor is reduced
#         self.t += 1
#         # TODO: popular decay heuristic from literature
#         self.beta *= self.beta_decay**self.t
#         if isinstance(x, list) and asscalar:
#             return gp_ucb.detach().numpy()
#         if asscalar:
#             gp_ucb = gp_ucb.detach().numpy().item()
#         return gp_ucb
#
#     def set_state(self, surrogate_model):
#         super().set_state(surrogate_model)
#
#         # Compute incumbent
#         if self.in_fill == "best":
#             # return torch.max(surrogate_model.y_)
#             self.incumbent = torch.min(self.surrogate_model.y_)
#         else:
#             x = self.surrogate_model.x
#             mu_train, _ = self.surrogate_model.predict(x)
#             # incumbent_idx = torch.argmax(mu_train)
#             incumbent_idx = torch.argmin(mu_train)
#             self.incumbent = self.surrogate_model.y_[incumbent_idx]
