from __future__ import annotations

from abc import abstractmethod


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
    def load_results(self, previous_results: dict, pending_evaluations: dict) -> None:
        raise NotImplementedError
