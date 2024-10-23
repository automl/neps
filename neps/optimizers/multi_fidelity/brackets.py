from __future__ import annotations

import logging
from collections.abc import Hashable, Sequence, Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from more_itertools import all_unique, pairwise

if TYPE_CHECKING:
    import pandas as pd
    from pandas import Index


logger = logging.getLogger(__name__)


@dataclass
class Rung(Sized):
    """A rung in a bracket"""

    value: int
    """The value of a rung, used to determine order between rungs."""

    table: pd.DataFrame
    """The slice of the table that constitutes this rung."""

    capacity: int | None
    """The capacity of the rung, if any."""

    def __len__(self) -> int:
        return len(self.table)

    @property
    def config_ids(self) -> list[int]:
        return self.table.index.get_level_values("id").unique().tolist()

    def has_pending(self) -> bool:
        return bool(self.table["perf"].isna().any())

    def has_capacity(self) -> bool:
        return self.capacity is None or len(self.table) < self.capacity

    def best_to_promote(
        self, *, exclude: Sequence[Hashable]
    ) -> tuple[int, dict[str, Any], float] | None:
        if exclude:
            contenders = self.table.drop(exclude)
            if contenders.empty:
                return None
        else:
            contenders = self.table

        best_ix, _best_rung = contenders["perf"].idxmin()  # type: ignore
        row = self.table.loc[(best_ix, _best_rung)]
        config = dict(row["config"])
        perf = row["perf"]
        return best_ix, config, perf

    def top_k(self, k: int) -> pd.DataFrame:
        return self.table.nsmallest(k, "perf")


@dataclass
class SyncBracket:
    """A bracket that holds a collection of rungs with a capacity constraint."""

    rungs: list[Rung]
    """A list of unique rungs, ordered from lowest to highest. The must have
    a capacity set.
    """

    def __post_init__(self) -> None:
        if not all_unique(rung.value for rung in self.rungs):
            raise ValueError(f"Got rungs with duplicate values\n{self.rungs}")

        if any(rung.value < 0 for rung in self.rungs):
            raise ValueError(f"Got rung with negative value\n{self.rungs}")

        if any(rung.capacity is None or rung.capacity < 1 for rung in self.rungs):
            raise ValueError(
                "All rungs must have a capacity set greater than 1"
                f"\nrungs: {len(self.rungs)}"
            )

        _sorted = sorted(self.rungs, key=lambda rung: rung.value)

        if any(
            lower.capacity < upper.capacity  # type: ignore
            for lower, upper in pairwise(_sorted)
        ):
            raise ValueError(f"Rungs must have a non-increasing capacity, got {_sorted}")

        self.rungs = _sorted

    def next(
        self,
    ) -> (
        tuple[Literal["promote"], dict, int, int]
        | tuple[Literal["new"], int]
        | Literal["pending", "done"]
    ):
        # If the bottom rung has capacity, we need to sample for it.
        bottom_rung = self.rungs[0]
        if bottom_rung.has_capacity():
            return "new", bottom_rung.value

        if not any(rung.has_capacity() for rung in self.rungs):
            return "done"

        lower, upper = next((l, u) for l, u in pairwise(self.rungs) if u.has_capacity())

        if lower.has_pending():
            return "pending"  # We need to wait before promoting

        promote_config = lower.best_to_promote(exclude=upper.config_ids)

        # If we have no promotable config, somehow the upper rung has more
        # capacity then lower. We check for this in the `__post_init__`
        if promote_config is None:
            raise RuntimeError(
                "This is a bug, either this bracket should have signified to have"
                " nothing promotable or pending"
            )

        _id, config, _perf = promote_config
        return "promote", config, _id, upper.value

    @classmethod
    def create_repeating_brackets(
        cls,
        table: pd.DataFrame,
        *,
        rung_sizes: dict[int, int],
    ) -> list[SyncBracket]:
        # Data has multi-index of (id, rung)
        all_ids = table.index.get_level_values("id").unique()

        # Split the ids into N brackets of size K.
        # K is the number of configurations in the lowest rung, i.e. number of config ids
        K = rung_sizes[min(rung_sizes)]
        N = len(all_ids) // K

        bracket_id_slices: list[Index] = [all_ids[i * K : (i + 1) * K] for i in range(N)]
        bracket_datas = [table.loc[bracket_ids] for bracket_ids in bracket_id_slices]

        # [bracket] -> {rung: table}
        data_for_bracket_by_rung = [
            dict(iter(d.groupby(level="rung", sort=False))) for d in bracket_datas
        ]

        # Used if there is nothing for one of the rungs
        empty_slice = table.loc[[]]

        return [
            SyncBracket(
                rungs=[
                    Rung(rung, data_by_rung.get(rung, empty_slice), capacity)
                    for rung, capacity in rung_sizes.items()
                ],
            )
            for data_by_rung in data_for_bracket_by_rung
        ]


@dataclass
class AsyncBracket:
    """A bracket that holds a collection of rungs with no capacity constraints."""

    rungs: list[Rung]
    """A list of rungs, ordered from lowest to highest."""

    eta: int
    """The eta parameter used for deciding when to promote.

    When any of the top_k configs in a rung can be promoted and have not been
    promoted yet, they will be.

    Here `k = len(rung) // eta`.
    """

    def __post_init__(self) -> None:
        self.rungs = sorted(self.rungs, key=lambda rung: rung.value)
        if any(rung.capacity is not None for rung in self.rungs):
            raise ValueError(
                "AsyncBracket was given a rung that has a capacity, however"
                " a rung in an async bracket should not have a capacity set."
                f"\nrungs: {self.rungs}"
            )

    def next(
        self,
    ) -> tuple[Literal["promote"], dict, int, int] | tuple[Literal["new"], int]:
        # Starting from the highest rung going down, check if any configs to promote
        for lower, upper in reversed(list(pairwise(self.rungs))):
            k = len(lower) // self.eta
            if k == 0:
                continue  # Not enough configs to promote yet

            best_k = lower.top_k(k)
            candidates = best_k.drop(upper.config_ids, errors="ignore")
            if candidates.empty:
                continue  # No configs that aren't already promoted

            promotable = candidates.iloc[0]
            _id, _rung = promotable.name
            config = dict(promotable["config"])
            return "promote", config, _id, upper.value

        # We couldn't find any promotions, sample at the lowest rung
        return "new", self.rungs[0].value

    @classmethod
    def make_asha_bracket(
        cls,
        table: pd.DataFrame,
        *,
        rungs: list[int],
        eta: int,
    ) -> AsyncBracket:
        return cls(
            rungs=[
                Rung(
                    rung,
                    capacity=None,
                    table=table.loc[table.index.get_level_values("rung") == rung],
                )
                for rung in rungs
            ],
            eta=eta,
        )
