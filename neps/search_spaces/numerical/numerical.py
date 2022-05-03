from abc import abstractmethod

from ..parameter import Parameter


class NumericalParameter(Parameter):
    def __init__(self, value=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    @property
    def id(self):
        return self.value

    @abstractmethod
    def _get_neighbours(self):
        raise NotImplementedError

    def serialize(self):
        return self.value

    def load_from(self, value):
        self.value = value
