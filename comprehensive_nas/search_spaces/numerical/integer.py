from typing import Union

import numpy as np

from .float import FloatHyperparameter


class IntegerHyperparameter(FloatHyperparameter):
    def __init__(
        self,
        name: str,
        lower: Union[float, int],
        upper: Union[float, int],
        log: bool = False,
    ):
        super().__init__(name, lower, upper, log)
        self.fhp = FloatHyperparameter(
            name=self.name,
            lower=self.lower - 0.499999,
            upper=self.upper + 0.499999,
            log=self.log,
        )
        self.value = None
        self._id = -1

    def __repr__(self):
        return "Integer {}-{:.07f}, range: [{}, {}], value: {}".format(
            self.name, self._id, self.lower, self.upper, self.value
        )

    def sample(self):
        self.fhp.sample()
        self.value = int(round(self.fhp.value))
        self._id = np.random.random()

    def mutate(
        self,
        parent=None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "local_search",
    ):
        mutant = self.fhp.mutate(
            parent=parent,
            mutation_rate=mutation_rate,
            mutation_strategy=mutation_strategy,
        )
        child = float_to_integer(mutant)
        return child

    def crossover(self, parent1, parent2=None):
        pass

    def _get_neighbours(self, std: float = 0.2, num_neighbours: int = 1):
        neighbours = self.fhp._get_neighbours(std, num_neighbours)
        for idx, neighbour in enumerate(neighbours):
            neighbours[idx] = float_to_integer(neighbour)
        return neighbours

    def _transform(self):
        self.fhp._transform()
        self.value = self.fhp.value

    def _inv_transform(self):
        self.fhp._inv_transform()
        self.value = int(round(self.fhp.value))


def float_to_integer(float_hp):
    int_hp = IntegerHyperparameter(
        name=float_hp.name,
        lower=int(round(float_hp.lower)),
        upper=int(round(float_hp.upper)),
        log=float_hp.log,
    )
    int_hp._id = float_hp._id
    int_hp.value = int(round(float_hp.value))

    return int_hp
