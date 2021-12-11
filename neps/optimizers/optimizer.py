from abc import abstractmethod


class Optimizer:
    @abstractmethod
    def new_result(self, job):
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        raise NotImplementedError
