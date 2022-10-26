from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PromotionPolicy(ABC):
    """Base class for implementing a sampling straregy for SH and its subclasses"""

    def __init__(self, eta: int):
        self.rung_members: dict = {}
        self.rung_members_performance: dict = {}
        self.rung_promotions: dict = {}
        self.eta = eta  # type: int
        self.max_rung: int = None

    def set_state(
        self,
        *,  # allows only keyword args
        max_rung: int,
        members: dict,
        performances: dict,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        self.max_rung = max_rung
        self.rung_members = members
        self.rung_members_performance = performances

    @abstractmethod
    def retrieve_promotions(self) -> dict:
        raise NotImplementedError


class SyncPromotionPolicy(PromotionPolicy):
    """Implements a synchronous promotion from lower to higher fidelity.

    Promotes only when all predefined number of config slots are full.
    """

    def __init__(self, eta, **kwargs):
        super().__init__(eta, **kwargs)
        self.config_map: dict = None
        self.rung_promotions = None

    def set_state(
        self,
        *,  # allows only keyword args
        max_rung: int,
        members: dict,
        performances: dict,
        config_map: dict,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:  # pylint: disable=unused-argument
        super().set_state(max_rung=max_rung, members=members, performances=performances)
        self.config_map = config_map

    def retrieve_promotions(self) -> dict:
        """Returns the top 1/eta configurations per rung if enough configurations seen"""
        assert self.config_map is not None

        self.rung_promotions = {rung: [] for rung in self.config_map.keys()}
        total_rung_evals = 0
        for rung in reversed(sorted(self.config_map.keys())):
            total_rung_evals += len(self.rung_members[rung])
            if (
                total_rung_evals >= self.config_map[rung]
                and np.isnan(self.rung_members_performance[rung]).sum()
            ):
                # if rung is full but incomplete evaluations, pause on promotions, wait
                return self.rung_promotions
            if rung == self.max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            if (
                total_rung_evals >= self.config_map[rung]
                and np.isnan(self.rung_members_performance[rung]).sum() == 0
            ):
                # if rung is full and no incomplete evaluations, find promotions
                top_k = (self.config_map[rung] // self.eta) - (
                    self.config_map[rung] - len(self.rung_members[rung])
                )
                selected_idx = np.argsort(self.rung_members_performance[rung])[:top_k]
                self.rung_promotions[rung] = self.rung_members[rung][selected_idx]
        return self.rung_promotions


class AsyncPromotionPolicy(PromotionPolicy):
    """Implements an asynchronous promotion from lower to higher fidelity.

    Promotes whenever a higher fidelity has at least eta configurations.
    """

    def __init__(self, eta, **kwargs):
        super().__init__(eta, **kwargs)

    def retrieve_promotions(self) -> dict:
        """Returns the top 1/eta configurations per rung if enough configurations seen"""
        for rung in range(self.max_rung + 1):
            if rung == self.max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            # if less than eta configurations seen, no promotions occur as top_k=0
            top_k = len(self.rung_members_performance[rung]) // self.eta
            _ordered_idx = np.argsort(self.rung_members_performance[rung])
            self.rung_promotions[rung] = np.array(self.rung_members[rung])[_ordered_idx][
                :top_k
            ].tolist()
        return self.rung_promotions
