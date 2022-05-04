from abc import abstractmethod
from copy import deepcopy


class Parameter:
    def __init__(self, is_fidelity=False, set_default_value=True):
        self.is_fidelity = is_fidelity
        if set_default_value:
            self.value = None

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def mutate(self, parent=None):
        raise NotImplementedError

    @abstractmethod
    def crossover(self, parent1, parent2=None):
        raise NotImplementedError

    @abstractmethod
    def serialize(self):
        raise NotImplementedError

    @abstractmethod
    def load_from(self, data):
        raise NotImplementedError

    def normalized(self):
        return deepcopy(self)

    def compute_prior(self):  # pylint: disable=no-self-use
        return 1

    def __eq__(self, other):
        # Assuming that two different classes should represent two different parameters
        return isinstance(other, self.__class__) and self.serialize() == other.serialize()
