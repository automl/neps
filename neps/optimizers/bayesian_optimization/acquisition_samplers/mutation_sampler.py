from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from more_itertools import first

from .base_acq_sampler import AcquisitionSampler
from .random_sampler import RandomSampler


def _propose_location(
    acquisition_function,
    candidates: list,
    top_n: int = 5,
    return_distinct: bool = True,
) -> tuple[Iterable, np.ndarray, np.ndarray]:
    """top_n: return the top n candidates wrt the acquisition function."""
    if return_distinct:
        eis = acquisition_function(candidates, asscalar=True)  # faster
        eis_, unique_idx = np.unique(eis, return_index=True)
        try:
            i = np.argpartition(eis_, -top_n)[-top_n:]
            indices = np.array([unique_idx[j] for j in i])
        except ValueError:
            eis = torch.tensor([acquisition_function(c) for c in candidates])
            _, indices = eis.topk(top_n)
    else:
        eis = torch.tensor([acquisition_function(c) for c in candidates])
        _, indices = eis.topk(top_n)
    xs = [candidates[int(i)] for i in indices]
    return xs, eis, indices


class MutationSampler(AcquisitionSampler):
    def __init__(
        self,
        pipeline_space,
        pool_size: int = 250,
        n_best: int = 10,
        mutate_size: int = None,
        allow_isomorphism: bool = False,
        check_isomorphism_history: bool = True,
        patience: int = 50,
    ):
        super().__init__(pipeline_space=pipeline_space, patience=patience)
        self.pool_size = pool_size
        self.n_best = n_best
        self.mutate_size = mutate_size
        self.allow_isomorphism = allow_isomorphism
        self.check_isomorphism_history = (
            check_isomorphism_history  # check for isomorphisms also in previous graphs
        )

        self.random_sampling = RandomSampler(
            pipeline_space=pipeline_space, patience=patience
        )

    def set_state(self, x, y) -> None:
        super().set_state(x, y)
        self.random_sampling.set_state(x, y)

    def sample(self, acquisition_function) -> tuple[list, list, np.ndarray]:
        return first(self.sample_batch(acquisition_function, 1))

    def sample_batch(self, acquisition_function, batch):
        pool = self.create_pool(acquisition_function, self.pool_size)

        samples, _, _ = _propose_location(
            acquisition_function=acquisition_function,
            top_n=batch,
            candidates=pool,
        )
        return samples

    def create_pool(self, acquisition_function, pool_size: int) -> list:
        if len(self.x) == 0:
            return self.random_sampling.sample_batch(acquisition_function, pool_size)

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
            remaining_patience = self.patience
            for _ in range(per_arch):
                while remaining_patience:
                    try:
                        # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                        child = config.mutate()
                    except Exception:
                        remaining_patience -= 1
                        continue

                    if not self.allow_isomorphism:
                        # if disallow isomorphism, we enforce that each time, we mutate n distinct graphs.
                        # For now we do not check the isomorphism in all of the previous graphs though
                        if child == config or child in evaluation_pool:
                            remaining_patience -= 1
                            continue

                    evaluation_pool.append(child)
                    break

        # Fill missing pool with random samples
        nrandom_archs = max(pool_size - len(evaluation_pool), 0)
        if nrandom_archs:
            random_evaluation_pool = self.random_sampling.sample_batch(
                acquisition_function, nrandom_archs
            )
            evaluation_pool += random_evaluation_pool

        return evaluation_pool
