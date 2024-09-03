from typing import Any
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from ifbo import FTPFN


class FTPFNSurrogate:
    """Special class to deal with PFN surrogate model and freeze-thaw acquisition."""

    def __init__(self, target_path: Path = None, version: str = "0.0.1", **kwargs):
        self.ftpfn = FTPFN(target_path=target_path, version=version)
        self.target_path = self.ftpfn.target_path
        self.version = self.ftpfn.version
        self.train_x = None
        self.train_y = None

    @property
    def device(self):
        return self.ftpfn.device
    
    def _get_logits(self, test_x: torch.Tensor) -> torch.Tensor:        
        return self.ftpfn.model(
            self._cast_tensor_shapes(self.train_x),
            self._cast_tensor_shapes(self.train_y),
            self._cast_tensor_shapes(test_x)
        )

    def _cast_tensor_shapes(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3 and x.shape[1] == 1:
            return x
        if len(x.shape) == 2:
            return x.reshape(x.shape[0], 1, x.shape[1])
        if len(x.shape) == 1:     
            return x.reshape(x.shape[0], 1)
        raise ValueError(f"Shape not recognized: {x.shape}")

    @torch.no_grad()
    def get_mean_performance(self, test_x: torch.Tensor) -> torch.Tensor:
        logits = self._get_logits(test_x).squeeze()
        return self.ftpfn.model.criterion.mean(logits)

    @torch.no_grad()
    def get_pi(self, test_x, y_best):
        logits = self._get_logits(test_x)
        return self.ftpfn.model.criterion.pi(
            logits.squeeze(), best_f=(1 - y_best).unsqueeze(1)
        )
    
    @torch.no_grad()
    def get_ei(self, test_x, y_best):
        logits = self._get_logits(test_x)
        return self.ftpfn.model.criterion.ei(
            logits.squeeze(), best_f=(1 - y_best).unsqueeze(1)
        )

    @torch.no_grad()
    def get_lcb(self, test_x, beta: float=(1-.682)/2):
        logits = self._get_logits(test_x)
        lcb = self.ftpfn.model.criterion.ucb(
            logits=logits,
            best_f=None,
            rest_prob=beta,
            maximize=False  # IMPORTANT to be False, should calculate the LCB using the lower-bound ICDF as per beta
        )
        return lcb
    
    @torch.no_grad()
    def get_ucb(self, test_x, beta: float=(1-.682)/2):
        logits = self._get_logits(test_x)
        lcb = self.ftpfn.model.criterion.ucb(
            logits=logits,
            best_f=None,
            rest_prob=beta,
            maximize=True  # IMPORTANT to be True, should calculate the UCB using the upper-bound ICDF as per beta
        )
        return lcb
