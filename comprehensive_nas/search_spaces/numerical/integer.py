from typing import List, Union
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
        return "Integer {}-{:.07f}, range: [{}, {}], value: {}".format(self.name, self._id, self.lower, self.upper, self.value)

    def to_integer(self):
        integer_hp = IntegerHyperparameter(name=self.name,
                                           lower=int(round(self.lower)),
                                           upper=int(round(self.upper)),
                                           log=self.log)
        integer_hp._id = self._id
        integer_hp.value = int(round(self.value))
        return integer_hp

    def sample(self):
        self.fhp.sample()
        self.value = int(round(self.fhp.value))
        self._id = np.random.random()

    def mutate(self,
               parent=None,
               mutation_rate: float = 1.0,
               mutation_strategy: str = "simple"):
        mutant = self.fhp.mutate(parent=parent, mutation_rate=mutation_rate, mutation_strategy=mutation_strategy)
        child = self.__copy__()
        child.value = int(round(mutant.value))
        child._id = mutant._id
        return child

    def crossover(self, parent1, parent2=None):
        pass
