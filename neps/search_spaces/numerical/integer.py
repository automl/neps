from __future__ import annotations

import math

from typing_extensions import Literal

from .float import FloatParameter


class IntegerParameter(FloatParameter):
    def __init__(
        self,
        lower: float | int,
        upper: float | int,
        log: bool = False,
        is_fidelity: bool = False,
        default: None | float | int = None,
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        super().__init__(lower, upper, log, is_fidelity, default, default_confidence)
        self.lower = int(math.ceil(self.lower))
        self.upper = int(math.floor(self.upper))
        # We subtract/add 0.499999 from lower/upper bounds respectively, such that
        # sampling in the float space gives equal probability for all integer values,
        # i.e. [x - 0.499999, x + 0.499999]
        self.float_hp = FloatParameter(
            lower=self.lower - 0.499999,
            upper=self.upper + 0.499999,
            log=self.log,
            is_fidelity=is_fidelity,
            default=default,
            default_confidence=default_confidence,
        )
        self.value: None | int = None

    def __repr__(self):
        return f"<Integer, range: [{self.lower}, {self.upper}], value: {self.value}>"

    def sample(self, user_priors: bool = False):
        self.float_hp.sample(user_priors=user_priors)
        self.value = int(round(self.float_hp.value))  # type: ignore[arg-type]

    def from_step(self, step, scale, in_place=True):
        value = super().from_step(step, scale)
        value = int(round(value))
        if in_place:
            self.value = value
        return value

    def mutate(
        self,
        parent=None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "local_search",
    ):
        if self.is_fidelity:
            raise ValueError("Trying to mutate fidelity param!")
        mutant = self.float_hp.mutate(
            parent=parent,
            mutation_rate=mutation_rate,
            mutation_strategy=mutation_strategy,
        )
        child = float_to_integer(mutant)
        return child

    def crossover(self, parent1, parent2=None):
        if self.is_fidelity:
            raise ValueError("Trying to crossover fidelity param!")
        if parent2 is None:
            parent2 = self

        proxy_self = self.copy()
        proxy_self.value = round((parent1.value + parent2.value) / 1)
        # pylint: disable=protected-access
        children = proxy_self._get_neighbours(std=0.1, num_neighbours=2)

        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        # expected len(children) == num_neighbours
        return children

    # pylint: disable=protected-access
    def _get_neighbours(self, std: float = 0.2, num_neighbours: int = 1):
        neighbours = self.float_hp._get_neighbours(std, num_neighbours)
        for idx, neighbour in enumerate(neighbours):
            neighbours[idx] = float_to_integer(neighbour)
        return neighbours

    def normalized(self):
        hp = FloatParameter(
            lower=self.lower,
            upper=self.upper,
            log=self.log,
            is_fidelity=self.is_fidelity,
            default=self.default,
        )
        hp.value = self.value
        return hp.normalized()


def float_to_integer(float_hp):
    int_hp = IntegerParameter(
        lower=int(round(float_hp.lower)),
        upper=int(round(float_hp.upper)),
        log=float_hp.log,
    )
    int_hp.value = int(round(float_hp.value))

    return int_hp
