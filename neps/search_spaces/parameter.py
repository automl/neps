from abc import abstractmethod


class Parameter:
    def __init__(self):
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
    def create_from_id(self, identifier):
        raise NotImplementedError

    def compute_prior(self):  # pylint: disable=no-self-use
        return 1
