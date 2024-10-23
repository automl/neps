from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PromotionPolicy(ABC):
    """Base class for implementing a sampling straregy for SH and its subclasses."""

    def __init__(self, eta: int):
        self.eta: int = eta

    @abstractmethod
    def retrieve_promotions(
        self,
        config_map: dict[int, int],
        rung_members: dict[int, list[Any]],
        rung_members_performance: dict[int, np.ndarray],
        max_rung: int,
    ) -> dict[int, list]:
        raise NotImplementedError


class SyncPromotionPolicy(PromotionPolicy):
    """Implements a synchronous promotion from lower to higher fidelity.

    Promotes only when all predefined number of config slots are full.
    """

    def __init__(self, eta: int):
        super().__init__(eta)

    def retrieve_promotions(
        self,
        config_map: dict[int, int],
        rung_members: dict[int, list[Any]],
        rung_members_performance: dict[int, np.ndarray],
        max_rung: int,
    ) -> dict[int, list]:
        """Returns the top 1/eta configurations per rung if enough configurations seen."""
        rung_promotions = {rung: [] for rung in config_map}
        total_rung_evals = 0
        for rung in sorted(config_map.keys(), reverse=True):
            total_rung_evals += len(rung_members[rung])

            # if rung is full but incomplete evaluations, pause on promotions, wait
            if (
                total_rung_evals >= config_map[rung]
                and np.isnan(rung_members_performance[rung]).sum()
            ):
                return rung_promotions

            # cease promotions for the highest rung (configs at max budget)
            if rung == max_rung:
                continue

            if (
                total_rung_evals >= config_map[rung]
                and np.isnan(rung_members_performance[rung]).sum() == 0
            ):
                # if rung is full and no incomplete evaluations, find promotions
                top_k = (config_map[rung] // self.eta) - (
                    config_map[rung] - len(rung_members[rung])
                )
                selected_idx = np.argsort(rung_members_performance[rung])[:top_k]
                rung_promotions[rung] = rung_members[rung][selected_idx]

        return rung_promotions


class AsyncPromotionPolicy(PromotionPolicy):
    """Implements an asynchronous promotion from lower to higher fidelity.

    Promotes whenever a higher fidelity has at least eta configurations.
    """

    def retrieve_promotions(
        self,
        config_map: dict[int, int],
        rung_members: dict[int, list[Any]],
        rung_members_performance: dict[int, np.ndarray],
        max_rung: int,
    ) -> dict[int, list]:
        """Returns the top 1/eta configurations per rung if enough configurations seen."""
        rung_promotions = {rung: [] for rung in config_map}

        for rung in range(max_rung + 1):
            # cease promotions for the highest rung (configs at max budget)
            if rung == max_rung:
                continue

            # if less than eta configurations seen, no promotions occur as top_k=0
            top_k = len(rung_members_performance[rung]) // self.eta
            _ordered_idx = np.argsort(rung_members_performance[rung])
            rung_promotions[rung] = np.array(rung_members[rung])[_ordered_idx][
                :top_k
            ].tolist()
        return rung_promotions
