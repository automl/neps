from typing import List, Union

from .float import FloatHyperparameter


class IntegerHyperparameter(FloatHyperparameter):
    def __init__(
        self,
        name: str,
        lower: Union[float, int],
        upper: Union[float, int],
        log: bool = False,
    ):
        super(IntegerHyperparameter, self).__init__(name, lower, upper, log)
        self.fhp = FloatHyperparameter(
            name=self.name,
            lower=self.lower - 0.499999,
            upper=self.upper + 0.499999,
            log=self.log,
        )

    def __repr__(self):
        return "Integer {}, range: [{}, {}]".format(self.name, self.lower, self.upper)

    def sample(self, random_state):
        return int(round(self.fhp.sample(random_state)))

    def mutate(self, parent=None):
        pass

    def crossover(self, parent1, parent2=None):
        pass
