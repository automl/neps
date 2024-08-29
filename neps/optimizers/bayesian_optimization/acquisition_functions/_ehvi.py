# from abc import ABC, abstractmethod
from itertools import product

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Module

# class MultiObjectiveBaseAcqusition(ABC):
#     def __init__(self, surrogate_models: dict):
#         self.surrogate_models = surrogate_models
#
#     def propose_location(self, *args):
#         """Propose new locations for subsequent sampling
#         This method should be overriden by respective acquisition function implementations."""
#         raise NotImplementedError
#
#     def optimize(self):
#         """This is the method that user should call for the Bayesian optimisation main loop."""
#         raise NotImplementedError
#
#     @abstractmethod
#     def eval(self, x, asscalar: bool = False):
#         """Evaluate the acquisition function at point x2. This should be overridden by respective acquisition
#         function implementations"""
#         raise NotImplementedError
#
#     def __call__(self, *args, **kwargs):
#         return self.eval(*args, **kwargs)
#
#     def reset_surrogate_model(self, surrogate_models: dict):
#         for objective, surrogate_model in surrogate_models.items():
#             self.surrogate_models[objective] = surrogate_model
#


class ExpectedHypervolumeImprovement(Module):  # , MultiObjectiveBaseAcqusition):
    def __init__(
        self,
        model,
        ref_point,
        partitioning,
    ) -> None:
        r"""Expected Hypervolume Improvement supporting m>=2 outcomes.

        Implementation from BOtorch, adapted from
        https://github.com/pytorch/botorch/blob/353f37649fa8d90d881e8ea20c11986b15723ef1/botorch/acquisition/multi_objective/analytic.py#L78

        This implements the computes EHVI using the algorithm from [Yang2019]_, but
        additionally computes gradients via auto-differentiation as proposed by
        [Daulton2020qehvi]_.

        Note: this is currently inefficient in two ways due to the binary partitioning
        algorithm that we use for the box decomposition:

            - We have more boxes in our decomposition
            - If we used a box decomposition that used `inf` as the upper bound for
                the last dimension *in all hypercells*, then we could reduce the number
                of terms we need to compute from 2^m to 2^(m-1). [Yang2019]_ do this
                by using DKLV17 and LKF17 for the box decomposition.

        TODO: Use DKLV17 and LKF17 for the box decomposition as in [Yang2019]_ for
        greater efficiency.

        TODO: Add support for outcome constraints.

        Example:
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> ref_point = [0.0, 0.0]
            >>> EHVI = ExpectedHypervolumeImprovement(model, ref_point, partitioning)
            >>> ehvi = EHVI(test_X)

        Args:
            model: A fitted model.
            ref_point: A list with `m` elements representing the reference point (in the
                outcome space) w.r.t. to which compute the hypervolume. This is a
                reference point for the objective values (i.e. after applying
                `objective` to the samples).
            partitioning: A `NondominatedPartitioning` module that provides the non-
                dominated front and a partitioning of the non-dominated space in hyper-
                rectangles.
            objective: An `AnalyticMultiOutputObjective`.
        """
        # TODO: we could refactor this __init__ logic into a
        # HypervolumeAcquisitionFunction Mixin
        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        better_than_ref = (partitioning.pareto_Y > ref_point).all(dim=1)
        if not better_than_ref.any() and partitioning.pareto_Y.shape[0] > 0:
            raise ValueError(
                "At least one pareto point must be better than the reference point."
            )
        super().__init__()
        self.model = model
        self.register_buffer("ref_point", ref_point)
        self.partitioning = partitioning
        cell_bounds = self.partitioning.get_hypercell_bounds()
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])
        # create indexing tensor of shape `2^m x m`
        self._cross_product_indices = torch.tensor(
            list(product(*[[0, 1] for _ in range(ref_point.shape[0])])),
            dtype=torch.long,
            device=ref_point.device,
        )
        self.normal = Normal(0, 1)

    def psi(self, lower: Tensor, upper: Tensor, mu: Tensor, sigma: Tensor) -> None:
        r"""Compute Psi function.

        For each cell i and outcome k:

            Psi(lower_{i,k}, upper_{i,k}, mu_k, sigma_k) = (
            sigma_k * PDF((upper_{i,k} - mu_k) / sigma_k) + (
            mu_k - lower_{i,k}
            ) * (1 - CDF(upper_{i,k} - mu_k) / sigma_k)
            )

        See Equation 19 in [Yang2019]_ for more details.

        Args:
            lower: A `num_cells x m`-dim tensor of lower cell bounds
            upper: A `num_cells x m`-dim tensor of upper cell bounds
            mu: A `batch_shape x 1 x m`-dim tensor of means
            sigma: A `batch_shape x 1 x m`-dim tensor of standard deviations (clamped).

        Returns:
            A `batch_shape x num_cells x m`-dim tensor of values.
        """
        u = (upper - mu) / sigma
        return sigma * self.normal.log_prob(u).exp() + (mu - lower) * (
            1 - self.normal.cdf(u)
        )

    def nu(self, lower: Tensor, upper: Tensor, mu: Tensor, sigma: Tensor) -> None:
        r"""Compute Nu function.

        For each cell i and outcome k:

            nu(lower_{i,k}, upper_{i,k}, mu_k, sigma_k) = (
            upper_{i,k} - lower_{i,k}
            ) * (1 - CDF((upper_{i,k} - mu_k) / sigma_k))

        See Equation 25 in [Yang2019]_ for more details.

        Args:
            lower: A `num_cells x m`-dim tensor of lower cell bounds
            upper: A `num_cells x m`-dim tensor of upper cell bounds
            mu: A `batch_shape x 1 x m`-dim tensor of means
            sigma: A `batch_shape x 1 x m`-dim tensor of standard deviations (clamped).

        Returns:
            A `batch_shape x num_cells x m`-dim tensor of values.
        """
        return (upper - lower) * (1 - self.normal.cdf((upper - mu) / sigma))

    def forward(self, X: Tensor) -> Tensor:
        posterior = [[_m.predict(_x) for _m in self.model] for _x in X]
        mu = torch.tensor([[_m[0].item() for _m in _p] for _p in posterior])[:, None, :]
        sigma = torch.tensor([[_s[1].item() for _s in _p] for _p in posterior])[
            :, None, :
        ]

        # clamp here, since upper_bounds will contain `inf`s, which
        # are not differentiable
        cell_upper_bounds = self.cell_upper_bounds.clamp_max(1e8)
        # Compute psi(lower_i, upper_i, mu_i, sigma_i) for i=0, ... m-2
        psi_lu = self.psi(
            lower=self.cell_lower_bounds, upper=cell_upper_bounds, mu=mu, sigma=sigma
        )
        # Compute psi(lower_m, lower_m, mu_m, sigma_m)
        psi_ll = self.psi(
            lower=self.cell_lower_bounds,
            upper=self.cell_lower_bounds,
            mu=mu,
            sigma=sigma,
        )
        # Compute nu(lower_m, upper_m, mu_m, sigma_m)
        nu = self.nu(
            lower=self.cell_lower_bounds, upper=cell_upper_bounds, mu=mu, sigma=sigma
        )
        # compute the difference psi_ll - psi_lu
        psi_diff = psi_ll - psi_lu

        # this is batch_shape x num_cells x 2 x (m-1)
        stacked_factors = torch.stack([psi_diff, nu], dim=-2)

        # Take the cross product of psi_diff and nu across all outcomes
        # e.g. for m = 2
        # for each batch and cell, compute
        # [psi_diff_0, psi_diff_1]
        # [nu_0, psi_diff_1]
        # [psi_diff_0, nu_1]
        # [nu_0, nu_1]
        # this tensor has shape: `batch_shape x num_cells x 2^m x m`
        all_factors_up_to_last = stacked_factors.gather(
            dim=-2,
            index=self._cross_product_indices.expand(
                stacked_factors.shape[:-2] + self._cross_product_indices.shape
            ),
        )
        # compute product for all 2^m terms,
        # sum across all terms and hypercells
        return all_factors_up_to_last.prod(dim=-1).sum(dim=-1).sum(dim=-1)
