from __future__ import annotations

import abc
import logging

import torch

from . import config_string

_logger = logging.getLogger(__name__)


class StringKernel(abc.ABC):
    def __init__(self):
        super().__init__()
        self.__name__ = self.__class__.__name__

    @staticmethod
    def normalize_gram(K: torch.Tensor):
        K_diag = torch.sqrt(torch.diag(K))
        K_diag_outer = torch.ger(K_diag, K_diag)
        return K / K_diag_outer

    @abc.abstractmethod
    def transform(
        self,
        configs: tuple[config_string.ConfigString],
    ):
        raise NotImplementedError()
