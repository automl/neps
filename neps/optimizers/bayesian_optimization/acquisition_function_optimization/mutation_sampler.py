from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from ..acquisition_functions.base_acquisition import BaseAcquisition
from .base_acq_optimizer import AcquisitionOptimizer
from .random_sampler import RandomSampler


def _propose_location(
    acquisition_function,
    candidates: list,
    top_n: int = 5,
    return_distinct: bool = True,
) -> tuple[Iterable, np.ndarray, np.ndarray]:
    """top_n: return the top n candidates wrt the acquisition function."""
    if return_distinct:
        eis = acquisition_function.eval(candidates, asscalar=True)  # faster
        eis_, unique_idx = np.unique(eis, return_index=True)
        try:
            i = np.argpartition(eis_, -top_n)[-top_n:]
            indices = np.array([unique_idx[j] for j in i])
        except ValueError:
            eis = torch.tensor([acquisition_function.eval(c) for c in candidates])
            _, indices = eis.topk(top_n)
    else:
        eis = torch.tensor([acquisition_function.eval(c) for c in candidates])
        _, indices = eis.topk(top_n)
    xs = [candidates[int(i)] for i in indices]
    return xs, eis, indices


class MutationSampler(AcquisitionOptimizer):
    def __init__(
        self,
        search_space,
        acquisition_function: BaseAcquisition,
        n_best: int = 10,
        mutate_size: int = None,
        allow_isomorphism: bool = False,
        check_isomorphism_history: bool = True,
        patience: int = 50,
    ):
        super().__init__(search_space, acquisition_function)
        self.n_best = n_best
        self.mutate_size = mutate_size
        self.allow_isomorphism = allow_isomorphism
        self.check_isomorphism_history = (
            check_isomorphism_history  # check for isomorphisms also in previous graphs
        )
        self.patience = patience

        self.random_sampling = RandomSampler(search_space)

    def sample(
        self, pool_size: int = 250, batch_size: int = 5
    ) -> tuple[list, list, np.ndarray]:
        pool = self.create_pool(pool_size)

        if batch_size is None:
            return pool
        if batch_size is not None and self.acquisition_function is None:
            raise Exception("Mutation sampler has no acquisition function!")

        samples, acq_vals, _ = _propose_location(
            acquisition_function=self.acquisition_function,
            top_n=batch_size,
            candidates=pool,
        )

        return samples, pool, acq_vals

    def create_pool(self, pool_size: int) -> list:
        if len(self.x) == 0:
            return self.random_sampling.sample(pool_size=pool_size)

        mutate_size = (
            int(0.5 * pool_size) if self.mutate_size is None else self.mutate_size
        )
        assert (
            pool_size >= mutate_size
        ), " pool_size must be larger or equal to mutate_size"

        n_best = len(self.x) if len(self.x) < self.n_best else self.n_best
        best_configs = [
            x for (_, x) in sorted(zip(self.y, self.x), key=lambda pair: pair[0])
        ][:n_best]
        evaluation_pool = []
        per_arch = mutate_size // n_best
        for config in best_configs:
            n_child = 0
            patience_ = self.patience
            while n_child < per_arch and patience_ > 0:
                try:
                    # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                    child = config.mutate()
                except Exception:
                    patience_ -= 1
                    continue

                if not self.allow_isomorphism:
                    # if disallow isomorphism, we enforce that each time, we mutate n distinct graphs. For now we do not
                    # check the isomorphism in all of the previous graphs though
                    if child == config or child in evaluation_pool:
                        patience_ -= 1
                        continue

                evaluation_pool.append(child)
                n_child += 1

        # Fill missing pool with random samples
        nrandom_archs = max(pool_size - len(evaluation_pool), 0)
        if nrandom_archs:
            random_evaluation_pool = self.random_sampling.sample(nrandom_archs)
            evaluation_pool += random_evaluation_pool

        return evaluation_pool
