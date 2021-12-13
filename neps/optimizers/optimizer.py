from abc import abstractmethod
from collections.abc import Iterable


class Optimizer:
    @abstractmethod
    def new_result(self, job):
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        raise NotImplementedError

    @abstractmethod
    def get_config_and_ids(self):
        raise NotImplementedError

    @abstractmethod
    def load_results(
        self, previous_results: Iterable, pending_evaluations: Iterable
    ) -> None:
        raise NotImplementedError
