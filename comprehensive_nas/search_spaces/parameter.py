from abc import abstractmethod


class Parameter:
    def __init__(self):
        pass

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def query(self, dataset_api, mode):
        raise NotImplementedError

    @abstractmethod
    def mutate(self, parent=None):
        raise NotImplementedError

    @abstractmethod
    def crossover(self, parent1, parent2=None):
        raise NotImplementedError
