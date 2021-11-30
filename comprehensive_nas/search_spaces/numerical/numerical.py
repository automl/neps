from abc import abstractmethod

from ..parameter import Parameter


class NumericalParameter(Parameter):
    def __init__(self):
        super().__init__()

    @property
    def id(self):
        return self.value

    @abstractmethod
    def _get_neighbours(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _transform(self):
        raise NotImplementedError

    @abstractmethod
    def _inv_transform(self):
        raise NotImplementedError
