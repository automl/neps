from typing import Tuple

import numpy as np

from ..acqusition_functions.base_acqusition import BaseAcquisition
from .base_acq_optimizer import AcquisitionOptimizer
from .random_sampler import RandomSampler


class MutationSampler(AcquisitionOptimizer):
    def __init__(
        self,
        search_space,
        acquisition_function: BaseAcquisition,
        n_best: int = 10,
        mutate_size: int = None,
        allow_isomorphism: bool = False,
        check_isomorphism_history: bool = True,  # on NB201 set to False!
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
    ) -> Tuple[list, list, np.ndarray]:
        pool = self.create_pool(pool_size)

        if batch_size is None:
            return pool
        if batch_size is not None and self.acquisition_function is None:
            raise Exception(f"Mutation sampler has no acquisition function!")

        samples, acq_vals, _ = self.acquisition_function.propose_location(
            top_n=batch_size, candidates=pool
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
        eval_pool_ids = (
            [x.id for x in self.x]
            if not self.allow_isomorphism and self.check_isomorphism_history
            else []
        )
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
                    if child.id == config.id:
                        patience_ -= 1
                        continue
                    if child.id in eval_pool_ids:
                        patience_ -= 1
                        continue

                evaluation_pool.append(child)
                eval_pool_ids.append(child.id)
                n_child += 1

        # Fill missing pool with random samples
        nrandom_archs = max(pool_size - len(evaluation_pool), 0)
        if nrandom_archs:
            random_evaluation_pool = self.random_sampling.sample(nrandom_archs)
            evaluation_pool += random_evaluation_pool

        return evaluation_pool
