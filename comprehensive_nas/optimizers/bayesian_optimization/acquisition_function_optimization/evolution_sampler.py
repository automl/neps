import random
from heapq import nlargest
from typing import Tuple

import numpy as np

from ..acqusition_functions.base_acqusition import BaseAcquisition
from .base_acq_optimizer import AcquisitionOptimizer
from .random_sampler import RandomSampler


class EvolutionSampler(AcquisitionOptimizer):
    def __init__(
        self,
        objective,
        acquisition_function: BaseAcquisition,
        num_evolutions: int = 10,
        p_tournament: float = 0.2,
        p_crossover: float = 0.5,
        p_self_crossover: float = 0.5,
        dynamic: bool = True,
        max_iters: int = 50,
        patience: int = 50,
        initial_history_last: int = 10,
        initial_history_best: int = 0,
        initial_history_acq_best: int = 0,
    ):
        # TODO implement isomorphism checking!
        super().__init__(objective, acquisition_function)
        self.num_evolutions = num_evolutions
        self.p_tournament = p_tournament
        self.p_crossover = p_crossover
        self.p_self_crossover = p_self_crossover
        self.dynamic = dynamic
        self.max_iters = max_iters
        self.patience = patience
        self.initial_history_last = initial_history_last
        self.initial_history_best = initial_history_best
        self.initial_history_acq_best = initial_history_acq_best

        self.random_sampling = RandomSampler(objective)

    def _mutate(self, parent):
        _patience = self.patience
        while _patience > 0:
            try:
                # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                child = parent.mutate()
                return child
            except Exception:
                _patience -= 1
                continue
        return False

    def _crossover(self, parent1, parent2):
        _patience = self.patience
        while _patience > 0:
            try:
                # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                children = parent1.crossover(parent2)
                return children
            except Exception:
                _patience -= 1
                continue
        return False

    def _tournament_selection(self, population: list, fitness: np.ndarray):
        size = int(len(population) * self.p_tournament)
        contender_indices = np.random.randint(len(population), size=size)
        contender_fitness = fitness[contender_indices]
        indices = nlargest(
            2, range(len(contender_fitness)), key=lambda idx: contender_fitness[idx]
        )
        return contender_indices[indices]

    def _evolve(self, population, fitness):
        new_pop = [None] * len(population)
        i = 0
        while i < len(population):
            # sample parent
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
                new_pop[i] = child1
                i += 1
                if i < len(population):
                    new_pop[i] = child2
                    i += 1
            else:
                child1 = self._mutate(parent1)
                if child1 is False:
                    continue
                new_pop[i] = child1
                i += 1
        return new_pop

    def evolution(
        self, previous_samples: list, population_size: int, batch_size: int = None
    ):
        def inner_loop(population, fitness, X_max, acq_max):
            try:
                fitness_standardized = fitness / np.sum(fitness)
            except Exception:
                fitness_standardized = 1 / len(fitness)
            population = self._evolve(population, fitness_standardized)
            # recalc fitness by also evaluating previous best configs
            fitness = self.acquisition_function.eval(X_max + population, asscalar=True)
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

        population = self.random_sampling.sample(population_size - len(previous_samples))
        population.extend(previous_samples)
        fitness = self.acquisition_function.eval(population, asscalar=True)

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

    def sample(
        self, pool_size: int = 200, batch_size: int = None
    ) -> Tuple[list, list, list]:
        if not batch_size < pool_size:
            raise Exception(
                f"Population size {pool_size} is smaller than batch size {batch_size}!"
            )
        population = self.x
        if self.initial_history_last > 0 and len(self.x) > self.initial_history_last:
            population = population[-self.initial_history_last :]
        if self.initial_history_best > 0 and len(self.x) > self.initial_history_best:
            indices = np.argpartition(self.y, self.initial_history_best)
            for idx in indices[: self.initial_history_best]:
                population.append(self.x[idx])
        if (
            self.initial_history_acq_best > 0
            and len(self.x) > self.initial_history_acq_best
        ):
            acq_vals = self.acquisition_function.eval(self.x, asscalar=True)
            indices = np.argpartition(acq_vals, -self.initial_history_acq_best)
            for idx in indices[-self.initial_history_acq_best :]:
                population.append(self.x[idx])
        return self.evolution(population, pool_size, batch_size)
