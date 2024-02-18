import random
from heapq import nlargest
from typing import List, Tuple

import numpy as np

from ....search_spaces.search_space import SearchSpace
from .base_acq_sampler import AcquisitionSampler
from .random_sampler import RandomSampler


class EvolutionSampler(AcquisitionSampler):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        pool_size: int = 200,
        num_evolutions: int = 10,
        p_tournament: float = 0.2,
        p_crossover: float = 0.5,
        p_self_crossover: float = 0.5,
        dynamic: bool = True,
        max_iters: int = 50,
        initial_history_best: int = 10,
        initial_history_last: int = 0,
        initial_history_acq_best: int = 0,
        allow_isomorphism: bool = True,
        check_isomorphism_history: bool = False,
        patience: int = 50,
    ):
        super().__init__(pipeline_space=pipeline_space, patience=patience)
        self.pool_size = pool_size
        self.num_evolutions = num_evolutions
        self.p_tournament = p_tournament
        self.p_crossover = p_crossover
        self.p_self_crossover = p_self_crossover
        self.dynamic = dynamic
        self.max_iters = max_iters
        self.initial_history_last = initial_history_last
        self.initial_history_best = initial_history_best
        self.initial_history_acq_best = initial_history_acq_best
        self.allow_isomorphism = allow_isomorphism
        # check for isomorphisms also in previous graphs
        self.check_isomorphism_history = check_isomorphism_history

        self.random_sampling = RandomSampler(
            pipeline_space=pipeline_space, patience=self.patience
        )

    def set_state(self, x, y) -> None:
        super().set_state(x, y)
        self.random_sampling.set_state(x, y)

    def _mutate(self, parent):
        for _ in range(self.patience):
            try:
                # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                return parent.mutate()
            except Exception:
                continue
        return False

    def _crossover(self, parent1, parent2):
        for _ in range(self.patience):
            try:
                # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                return parent1.crossover(parent2)
            except Exception:
                continue
        return False, False

    def _tournament_selection(self, population: list, fitness: np.ndarray):
        size = int(len(population) * self.p_tournament)
        contender_indices = np.random.randint(len(population), size=size)
        contender_fitness = fitness[contender_indices]
        indices = nlargest(
            2, range(len(contender_fitness)), key=lambda idx: contender_fitness[idx]
        )
        return contender_indices[indices]

    def _evolve(self, population, fitness):
        new_pop = []
        while len(new_pop) < len(population):
            # sample parents
            best, second_best = self._tournament_selection(population, fitness)
            parent1 = population[best]
            parent2 = population[second_best]

            if random.random() < self.p_crossover:
                if random.random() < self.p_self_crossover:
                    child1, child2 = self._crossover(parent1, parent1)
                else:
                    child1, child2 = self._crossover(parent1, parent2)
                if child1 is False:
                    continue
                if not self.allow_isomorphism and child1 in new_pop:
                    continue
                new_pop.append(child1)
                if len(new_pop) < len(population):
                    if not self.allow_isomorphism and child2 in new_pop:
                        continue
                    new_pop.append(child2)
            else:
                child1 = self._mutate(parent1)
                if child1 is False:
                    continue
                if not self.allow_isomorphism and child1 in new_pop:
                    continue
                new_pop.append(child1)
        return new_pop

    def evolution(
        self,
        acquisition_function,
        previous_samples: list,
        population_size: int,
        batch_size: int = None,
    ):
        def inner_loop(population, fitness, X_max, acq_max):
            try:
                fitness_standardized = fitness / np.sum(fitness)
            except Exception:
                fitness_standardized = 1 / len(fitness)
            population = self._evolve(population, fitness_standardized)
            # recalc fitness by also evaluating previous best configs
            fitness = acquisition_function(X_max + population, asscalar=True)
            # update best config & score
            indices = nlargest(
                batch_size, range(len(fitness)), key=lambda idx: fitness[idx]
            )
            X_max = [
                population[idx - batch_size] if idx >= batch_size else X_max[idx]
                for idx in indices
            ]
            acq_max = [float(fitness[idx]) for idx in indices]
            return population, fitness, X_max, acq_max

        if batch_size is None:
            batch_size = 1

        new_pop = (
            self.x
            if not self.allow_isomorphism and self.check_isomorphism_history
            else []
        )
        population: List[SearchSpace] = []
        remaining_patience = self.patience
        while (
            population_size - len(previous_samples) > len(population)
            and remaining_patience > 0
        ):
            population.extend(
                [
                    p_member
                    for p_member in self.random_sampling.sample_batch(
                        acquisition_function,
                        population_size - len(previous_samples) - len(population),
                    )
                    if p_member not in new_pop
                ]
            )
            remaining_patience -= 1
        if (
            remaining_patience == 0
            and (population_size - len(previous_samples)) - len(population) > 0
        ):
            population += self.random_sampling.sample(
                population_size - len(previous_samples) - len(population)
            )
        population.extend(previous_samples)
        fitness = acquisition_function(population, asscalar=True)

        # initialize best config & score
        indices = nlargest(batch_size, range(len(fitness)), key=lambda idx: fitness[idx])
        X_max = [population[idx] for idx in indices]
        acq_max = [fitness[idx] for idx in indices]
        iterations_best = [acq_max]
        for _ in range(self.num_evolutions):
            population, fitness, X_max, acq_max = inner_loop(
                population, fitness, X_max, acq_max
            )
            iterations_best.append(acq_max)
        if self.dynamic:
            i = self.num_evolutions
            while i < self.max_iters:
                population, fitness, X_max, acq_max = inner_loop(
                    population, fitness, X_max, acq_max
                )
                if all(
                    all(np.isclose(x, l) for l in list(zip(*iterations_best[-5:]))[j])
                    for j, x in enumerate(acq_max)
                ):
                    break
                iterations_best.append(acq_max)
                i += 1

        return X_max, population, acq_max

    def sample(self, acquisition_function) -> Tuple[list, list, list]:
        population: List[SearchSpace] = []
        if self.initial_history_last > 0 and len(self.x) >= self.initial_history_last:
            population = self.x[-self.initial_history_last :]
        if self.initial_history_best > 0 and len(self.x) >= self.initial_history_best:
            if len(self.y) > self.initial_history_best:
                indices = np.argpartition(self.y, self.initial_history_best)
            else:
                indices = list(range(len(self.y)))
            for idx in indices[: self.initial_history_best]:
                population.append(self.x[idx])
        if (
            self.initial_history_acq_best > 0
            and len(self.x) >= self.initial_history_acq_best
        ):
            acq_vals = acquisition_function(self.x, asscalar=True)
            indices = np.argpartition(acq_vals, -self.initial_history_acq_best)
            for idx in indices[-self.initial_history_acq_best :]:
                population.append(self.x[idx])
        if (
            len(population)
            < self.initial_history_last
            + self.initial_history_best
            + self.initial_history_acq_best
        ):
            population += list(
                random.sample(
                    self.x,
                    k=min(
                        self.initial_history_last
                        + self.initial_history_best
                        + self.initial_history_acq_best
                        - len(population),
                        len(self.x),
                    ),
                )
            )
        X_max, _, _ = self.evolution(acquisition_function, population, self.pool_size, 1)
        return X_max[0]
