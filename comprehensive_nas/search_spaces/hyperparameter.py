from abc import abstractmethod


class Hyperparameter:
    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError("Hyperparameter's name should be a string.")
        self.name = name

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
    def _get_neighbours(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _transform(self):
        raise NotImplementedError

    @abstractmethod
    def _inv_transform(self):
        raise NotImplementedError
