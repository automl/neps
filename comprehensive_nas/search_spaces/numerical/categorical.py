from typing import Union, List
import numpy as np

from ..hyperparameter import Hyperparameter


class CategoricalHyperparameter(Hyperparameter):
    def __init__(self, name: str, choices: List[Union[float, int, str]]):
        super(CategoricalHyperparameter, self).__init__(name)
        self.choices = list(choices)
        self.num_choices = len(self.choices)
        self.probabilities = np.ones(self.num_choices) * (1./self.num_choices)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name and
            self.choices == other.choices
        )

    def __hash__(self):
        return hash(
            (
                self.name,
                self.choices
            )
        )

    def __repr__(self):
        return "Categorical {}, choices: {}".format(
            self.name, self.choices
        )

    def __copy__(self):
        return self.__class__(
            name=self.name,
            choices=self.choices
        )

    def sample(self, random_state):
        idx = random_state.choice(a=self.num_choices, replace=True, p=self.probabilities)
        return self.choices[int(idx)]

    def mutate(self, parent=None):
        pass

    def crossover(self, parent1, parent2=None):
        pass
