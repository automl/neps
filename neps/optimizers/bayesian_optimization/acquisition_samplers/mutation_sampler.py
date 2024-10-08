from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
import torch
from more_itertools import first

from neps.optimizers.bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)
from neps.optimizers.bayesian_optimization.acquisition_samplers.random_sampler import (
    RandomSampler,
)

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace


def _propose_location(
    *,
    acquisition_function: Callable,
    candidates: list[SearchSpace],
    top_n: int = 5,
    return_distinct: bool = True,
) -> tuple[list[SearchSpace], np.ndarray | torch.Tensor, np.ndarray]:
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
    return xs, eis, np.asarray(indices)


class MutationSampler(AcquisitionSampler):
    def __init__(
        self,
        *,
        pipeline_space: SearchSpace,
        pool_size: int = 250,
        n_best: int = 10,
        mutate_size: float | int = 0.5,
        allow_isomorphism: bool = False,
        check_isomorphism_history: bool = True,
        patience: int = 50,
    ):
        super().__init__(pipeline_space=pipeline_space, patience=patience)
        self.pool_size = pool_size
        self.n_best = n_best
        self.mutate_size = mutate_size
        if isinstance(mutate_size, int):
            assert (
                pool_size >= mutate_size
            ), " pool_size must be larger or equal to mutate_size"

        self.allow_isomorphism = allow_isomorphism
        self.check_isomorphism_history = (
            check_isomorphism_history  # check for isomorphisms also in previous graphs
        )

        self.random_sampling = RandomSampler(
            pipeline_space=pipeline_space, patience=patience
        )

    @override
    def set_state(
        self, x: list[SearchSpace], y: Sequence[float] | np.ndarray | torch.Tensor
    ) -> None:
        super().set_state(x, y)
        self.random_sampling.set_state(x, y)

    @override
    def sample(self, acquisition_function: Callable) -> SearchSpace:
        return first(self.sample_batch(acquisition_function, batch=1))

    @override
    def sample_batch(
        self,
        acquisition_function: Callable,
        batch: int,
    ) -> list[SearchSpace]:
        pool = self.create_pool(
            x=self.x,
            y=self.y,
            acquisition_function=acquisition_function,
            pool_size=self.pool_size,
        )

        samples, _, _ = _propose_location(
            acquisition_function=acquisition_function,
            top_n=batch,
            candidates=pool,
        )
        return samples

    def create_pool(
        self,
        x: list[SearchSpace],
        y: Sequence[float] | np.ndarray | torch.Tensor,
        acquisition_function: Callable,
        pool_size: int,
    ) -> list[SearchSpace]:
        if len(x) == 0:
            return self.random_sampling.sample_batch(acquisition_function, pool_size)

        if isinstance(self.mutate_size, int):
            mutate_size = self.mutate_size
        else:
            mutate_size = int(self.mutate_size * pool_size)

        n_best = len(self.x) if len(self.x) < self.n_best else self.n_best
        best_configs = [
            x
            for (_, x) in sorted(
                zip(self.y, self.x, strict=False),
                key=lambda pair: pair[0],
            )
        ][:n_best]

        seen: set[int] = set()

        def _hash(_config: SearchSpace) -> int:
            return hash(_config.hp_values().values())

        evaluation_pool = []
        per_arch = mutate_size // n_best

        for config in best_configs:
            remaining_patience = self.patience
            for _ in range(per_arch):
                while remaining_patience:
                    try:
                        # needs to throw an Exception if config is not valid,
                        # e.g., empty graph etc.!
                        child = config.mutate()
                    except Exception:  # noqa: BLE001
                        remaining_patience -= 1
                        continue
                    hash_child = _hash(child)

                    # if disallow isomorphism, we enforce that each time,
                    # we mutate n distinct graphs. For now we do not check
                    # the isomorphism in all of the previous graphs though
                    if not self.allow_isomorphism and (
                        child == config or hash_child in seen
                    ):
                        remaining_patience -= 1
                        continue

                    evaluation_pool.append(child)
                    seen.add(hash_child)
                    break

        # Fill missing pool with random samples
        nrandom_archs = max(pool_size - len(evaluation_pool), 0)
        if nrandom_archs:
            random_evaluation_pool = self.random_sampling.sample_batch(
                acquisition_function, nrandom_archs
            )
            evaluation_pool += random_evaluation_pool

        return evaluation_pool
