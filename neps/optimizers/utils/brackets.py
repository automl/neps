from __future__ import annotations

import logging
from collections.abc import Hashable, Sequence, Sized
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from more_itertools import all_unique, pairwise

if TYPE_CHECKING:
    import pandas as pd
    from pandas import Index


logger = logging.getLogger(__name__)


@dataclass
class PromoteAction:
    config: dict[str, Any]
    id: int
    new_rung: int


@dataclass
class SampleAction:
    rung: int


BracketAction: TypeAlias = PromoteAction | SampleAction | Literal["pending", "done"]


def calculate_sh_rungs(
    bounds: tuple[int, int] | tuple[float, float],
    eta: int,
    early_stopping_rate: int,
) -> tuple[dict[int, int | float], dict[int, int]]:
    bmin, bmax = bounds
    budget_type = int if isinstance(bmin, int) else float
    esr = early_stopping_rate
    stop_rate_limit = int(np.floor(np.log(bmax / bmin) / np.log(eta)))
    assert esr <= stop_rate_limit

    nrungs = int(np.floor(np.log(bmax / (bmin * (eta**esr))) / np.log(eta)) + 1)
    rung_to_fidelity = {
        esr + j: budget_type(bmax / (eta**i))
        for i, j in enumerate(reversed(range(nrungs)))
    }

    # L2 from Alg 1 in https://arxiv.org/pdf/1603.06560.pdf
    s_max = stop_rate_limit + 1
    _s = stop_rate_limit - esr
    _n_config = int(np.floor(s_max / (_s + 1)) * eta**_s)
    rung_sizes = {i + esr: _n_config // (eta**i) for i in range(nrungs)}
    return rung_to_fidelity, rung_sizes


def calculate_hb_bracket_layouts(
    bounds: tuple[int, int] | tuple[float, float],
    eta: int,
) -> tuple[dict[int, int | float], list[dict[int, int]]]:
    bmin, bmax = bounds
    budget_type = int if isinstance(bmin, int) else float
    stop_rate_limit = int(np.floor(np.log(bmax / bmin) / np.log(eta)))

    nrungs = int(np.floor(np.log(bmax / bmin) / np.log(eta))) + 1
    rung_to_fidelity = {
        j: budget_type(bmax / (eta**i)) for i, j in enumerate(reversed(range(nrungs)))
    }

    # L2 from Alg 1 in https://arxiv.org/pdf/1603.06560.pdf
    bracket_layouts: list[dict[int, int]] = []
    s_max = stop_rate_limit + 1
    for esr in range(nrungs):
        _s = stop_rate_limit - esr
        _n_config = int(np.floor(s_max / (_s + 1)) * eta**_s)

        sh_rungs = int(np.floor(np.log(bmax / (bmin * (eta**esr))) / np.log(eta)) + 1)
        rung_sizes = {i + esr: _n_config // (eta**i) for i in range(sh_rungs)}
        bracket_layouts.append(rung_sizes)

    return rung_to_fidelity, bracket_layouts


def async_hb_sample_bracket_to_run(max_rung: int, eta: int) -> int:
    # Sampling distribution derived from Appendix A (https://arxiv.org/abs/2003.10865)
    # Adapting the distribution based on the current optimization state
    # s \in [0, max_rung] and to with the denominator's constraint, we have K > s - 1
    # and thus K \in [1, ..., max_rung, ...]
    # Since in this version, we see the full SH rung, we fix the K to max_rung
    K = max_rung
    bracket_probs = [eta ** (K - s) * (K + 1) / (K - s + 1) for s in range(max_rung + 1)]
    bracket_probs = np.array(bracket_probs) / sum(bracket_probs)
    return int(np.random.choice(range(max_rung + 1), p=bracket_probs))


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
        return self.table.index.get_level_values("id").unique().tolist()  # type: ignore

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

    def pareto_promotion_sync(
        self,
        *,
        k: int,
        exclude: Sequence[Hashable] = [],
        mo_selector: Literal["nsga2", "epsnet"] = "epsnet",
    ) -> tuple[int, dict[str, Any], float] | None:
        """Selects the best configurations based on Pareto front for
        sync bracket optimizers.
        """
        if exclude:
            contenders = self.table.drop(exclude)
            if contenders.empty:
                return None
        else:
            contenders = self.table
        _df = self.mo_selector(
            selector=mo_selector,
            contenders=contenders,
            k=k,
        )
        _idx, _rung = _df.index[0]
        row = _df.loc[(_idx, _rung)]
        config = dict(row["config"])
        perf = row["perf"]
        return _idx, config, perf

    def top_k(self, k: int) -> pd.DataFrame:
        return self.table.nsmallest(k, "perf")

    def mo_selector(
        self,
        *,
        selector: Literal["nsga2", "epsnet"] = "epsnet",
        contenders: pd.DataFrame | None = None,
        k: int,
    ) -> pd.DataFrame:
        """Replaces top_k in single objective Bracket Optimizers
        with a multi-objective selector, which selects the best
        configurations based on the Pareto front.
        Page 4, Algorithm 2, Line 11: `mo_selector` in the MO-ASHA paper:
        https://arxiv.org/pdf/2106.12639
        """
        if contenders is None:
            contenders = self.table
        match selector:
            case "nsga2":
                raise NotImplementedError(
                    "NSGA2 selector is not implemented yet. Please use epsnet."
                )
            case "epsnet":
                return self.epsnet_selector(
                    k=k,
                    contenders=contenders,
                )
            case _:
                raise ValueError(
                    f"Unknown selector {selector}, please use either nsga2 or epsnet"
                )

    def nsga2_selector(
        self,
        k: int,
    ) -> pd.DataFrame:
        """Selects the best configurations based on NSGA2 algorithm.
        Uses Non-dominated sorting and Crowding distance from Pymoo.
        """

    def epsnet_selector(
        self,
        *,
        k: int,
        contenders: pd.DataFrame,
    ) -> pd.DataFrame:
        """Selects the best configurations based on epsilon-net sorting strategy.
        Uses Epsilon-net based sorting from SyneTune.
        """
        from neps.optimizers.utils.multiobjective.epsnet import nondominated_sort

        mo_costs = np.vstack(contenders["perf"].values)
        indices = nondominated_sort(
            X=mo_costs,
            max_items=k,
        )
        return contenders.iloc[indices]


@dataclass
class Sync:
    """A bracket that holds a collection of rungs with a capacity constraint."""

    rungs: list[Rung]
    """A list of unique rungs, ordered from lowest to highest. The must have
    a capacity set.
    """

    is_multi_objective: bool = field(default=False)
    """Whether the BracketOptimizer is multi-objective or not."""

    mo_selector: Literal["nsga2", "epsnet"] = field(default="epsnet")
    """The selector to use for multi-objective optimization."""

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

    def next(self) -> BracketAction:
        bottom_rung = self.rungs[0]

        # If the bottom rung has capacity, we need to sample for it.
        if bottom_rung.has_capacity():
            return SampleAction(bottom_rung.value)

        if not any(rung.has_capacity() for rung in self.rungs):
            return "done"

        lower, upper = next((l, u) for l, u in pairwise(self.rungs) if u.has_capacity())

        if lower.has_pending():
            return "pending"  # We need to wait before promoting

        if self.is_multi_objective:
            promote_config = lower.pareto_promotion_sync(
                mo_selector=self.mo_selector,
                k=1,
                exclude=upper.config_ids,
            )
        else:
            promote_config = lower.best_to_promote(exclude=upper.config_ids)

        # If we have no promotable config, somehow the upper rung has more
        # capacity then lower. We check for this in the `__post_init__`
        if promote_config is None:
            raise RuntimeError(
                "This is a bug, either this bracket should have signified to have"
                " nothing promotable or pending"
            )

        _id, config, _perf = promote_config
        return PromoteAction(config, _id, upper.value)

    @classmethod
    def create_repeating(
        cls,
        table: pd.DataFrame,
        *,
        rung_sizes: dict[int, int],
        is_multi_objective: bool = False,
        mo_selector: Literal["nsga2", "epsnet"] = "epsnet",
    ) -> list[Sync]:
        """Create a list of brackets from the table.

        The table should have a multi-index of (id, rung) where rung is the
        fidelity level of the configuration.

        This method will always ensure there is at least one bracket, with at least one
        empty slot. For example, if each bracket houses a maximum of 9 configurations,
        and there are 27 total unique configurations in the table, these will be split
        into 3 brackets with 9 configurations + 1 additional bracket with 0 in it
        configurations.

        ```
        # Unrealistic example showing the format of the table
        (id, rung) -> config, perf
        --------------------------
        0     0   |  {"hp": 0, ...}, 0.1
              1   |  {"hp": 0, ...}, 0.2
        1     0   |  {"hp": 1, ...}, 0.1
        2     1   |  {"hp": 2, ...}, 0.3
              2   |  {"hp": 2, ...}, 0.4
        3     2   |  {"hp": 3, ...}, 0.4
        ```

        Args:
            table: The table of configurations to split into brackets.
            rung_sizes: A mapping of rung to the capacity of that rung.

        Returns:
            Brackets which have each subselected the table with the corresponding rung
            sizes.
        """
        # Split the trials by their unique_id, taking batches of K at a time, which will
        # gives us N = len(unique_is) / K brackets in total.
        #
        # Here, unique_id referes to the `1` in config_1_0 i.e. id = 1, rung = 0
        #
        #      1  2  3  4  5  6  7  8  9   10 11 12 13 14 15 16 17 18   ...
        #     |           bracket1       |       bracket 2            | ... |

        # K is the number of configurations in the lowest rung, which is how many unique
        # ids are needed to fill a single bracket.
        K = rung_sizes[min(rung_sizes)]

        # N is the number of brackets we need to create to accomodate all the unique ids.
        # First we need all of the unique ids.
        uniq_ids = table.index.get_level_values("id").unique()

        # The formula (len(uniq_ids) + K) // K is used instead of
        # len(uniq_ids) // K. reason: make to ensure that even if the number of
        # unique IDs is less than K, at least one bracket is created
        N = (len(uniq_ids) + K) // K

        # Now we take the unique ids and split them into batches of size K
        bracket_id_slices: list[Index] = [uniq_ids[i * K : (i + 1) * K] for i in range(N)]

        # And now select the data for each of the unique_ids in the bracket
        bracket_datas = [
            table.loc[bracket_unique_ids] for bracket_unique_ids in bracket_id_slices
        ]

        # This will give us a list of dictionaries, where each element `n` of the
        # list is on of the `N` brackets, and the dictionary at element `n` maps
        # from a rung, to the slice of the data for that rung.
        all_N_bracket_datas = [
            dict(iter(d.groupby(level="rung", sort=False))) for d in bracket_datas
        ]

        # Used if there is nothing for one of the rungs
        empty_slice = table.loc[[]]

        return [
            Sync(
                rungs=[
                    Rung(rung, bracket_data.get(rung, empty_slice), capacity)
                    for rung, capacity in rung_sizes.items()
                ],
                is_multi_objective=is_multi_objective,
                mo_selector=mo_selector,
            )
            for bracket_data in all_N_bracket_datas
        ]


@dataclass
class Async:
    """A bracket that holds a collection of rungs with no capacity constraints."""

    rungs: list[Rung]
    """A list of rungs, ordered from lowest to highest."""

    eta: int
    """The eta parameter used for deciding when to promote.

    When any of the top_k configs in a rung can be promoted and have not been
    promoted yet, they will be.

    Here `k = len(rung) // eta`.
    """

    is_multi_objective: bool = field(default=False)
    """Whether the BracketOptimizer is multi-objective or not."""

    mo_selector: Literal["nsga2", "epsnet"] = field(default="epsnet")
    """The selector to use for multi-objective optimization."""

    def __post_init__(self) -> None:
        self.rungs = sorted(self.rungs, key=lambda rung: rung.value)
        if any(rung.capacity is not None for rung in self.rungs):
            raise ValueError(
                "AsyncBracket was given a rung that has a capacity, however"
                " a rung in an async bracket should not have a capacity set."
                f"\nrungs: {self.rungs}"
            )

    def next(self) -> BracketAction:
        # Starting from the highest rung going down, check if any configs to promote
        for lower, upper in reversed(list(pairwise(self.rungs))):
            import copy

            lower_dropped = copy.deepcopy(lower)
            lower_dropped.table = lower_dropped.table.drop(
                upper.config_ids,
                axis="index",
                level="id",
                errors="ignore",
            )
            k = len(lower_dropped) // self.eta
            if k == 0:
                continue  # Not enough configs to promote yet

            if self.is_multi_objective:
                best_k = lower_dropped.mo_selector(selector=self.mo_selector, k=k)
            else:
                best_k = lower_dropped.top_k(k)
            candidates = best_k.copy(deep=True)
            if candidates.empty:
                continue  # No configs that aren't already promoted

            promotable = candidates.iloc[0]
            _id, _rung = promotable.name
            config = dict(promotable["config"])
            return PromoteAction(config, _id, upper.value)

        # We couldn't find any promotions, sample at the lowest rung
        return SampleAction(self.rungs[0].value)

    @classmethod
    def create(
        cls,
        table: pd.DataFrame,
        *,
        rungs: list[int],
        eta: int,
        is_multi_objective: bool = False,
        mo_selector: Literal["nsga2", "epsnet"] = "epsnet",
    ) -> Async:
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
            is_multi_objective=is_multi_objective,
            mo_selector=mo_selector,
        )


@dataclass
class Hyperband:
    sh_brackets: list[Sync]

    _min_rung: int = field(init=False, repr=False)
    _max_rung: int = field(init=False, repr=False)

    is_multi_objective: bool = field(default=False)
    mo_selector: Literal["nsga2", "epsnet"] = field(default="epsnet")

    def __post_init__(self) -> None:
        if not self.sh_brackets:
            raise ValueError("HyperbandBrackets must have at least one SH bracket")

        # Sort the brackets by those which contain the lowest rung values first
        self.sh_brackets = sorted(
            self.sh_brackets, key=lambda sh_bracket: sh_bracket.rungs[0].value
        )
        self._min_rung = min(bracket.rungs[0].value for bracket in self.sh_brackets)
        self._max_rung = max(bracket.rungs[-1].value for bracket in self.sh_brackets)

    @classmethod
    def create_repeating(
        cls,
        table: pd.DataFrame,
        *,
        bracket_layouts: list[dict[int, int]],
        is_multi_objective: bool = False,
        mo_selector: Literal["nsga2", "epsnet"] = "epsnet",
    ) -> list[Hyperband]:
        """Create a list of brackets from the table.

        The table should have a multi-index of (id, rung) where rung is the
        fidelity level of the configuration.

        This method will always ensure there is at least one hyperband set of brackets,
        with at least one empty slot. For example, if each hyperband set of brackets
        houses a maximum of 9 configurations, and there are 27 total unique configurations
        in the table, these will be split into 3 hyperband brackets with 9 configurations
        + 1 additional hyperband bracket with 0 in it configurations.

        ```
        # Unrealistic example showing the format of the table
        (id, rung) -> config, perf
        --------------------------
        0     0   |  {"hp": 0, ...}, 0.1
              1   |  {"hp": 0, ...}, 0.2
        1     0   |  {"hp": 1, ...}, 0.1
        2     1   |  {"hp": 2, ...}, 0.3
              2   |  {"hp": 2, ...}, 0.4
        3     2   |  {"hp": 3, ...}, 0.4
        ```

        Args:
            table: The table of configurations to split into brackets.
            bracket_layouts: A mapping of rung to the capacity of that rung.

        Returns:
            HyperbandBrackets which have each subselected the table with the
            corresponding rung sizes.
        """
        all_ids = table.index.get_level_values("id").unique()

        # Split the ids into N hyperband brackets of size K.
        # K is sum of number of configurations in the lowest rung of each SH bracket
        #
        # For example:
        # > bracket_layouts = [
        # >   {0: 81, 1: 27, 2: 9, 3: 3, 4: 1},
        # >   {1: 27, 2: 9, 3: 3, 4: 1},
        # >   {2: 9, 3: 3, 4: 1},
        # >   ...
        # > ]
        #
        # Corresponds to:
        # bracket1 - [rung_0: 81, rung_1: 27, rung_2: 9, rung_3: 3, rung_4: 1]
        # bracket2 - [rung_1: 27, rung_2: 9, rung_3: 3, rung_4: 1]
        # bracket3 - [rung_2: 9, rung_3: 3, rung_4: 1]
        # ...
        # > K = 81 + 27 + 9 + ...
        #
        bottom_rung_sizes = [sh[min(sh.keys())] for sh in bracket_layouts]
        K = sum(bottom_rung_sizes)
        N = max(len(all_ids) // K + 1, 1)

        hb_id_slices: list[Index] = [all_ids[i * K : (i + 1) * K] for i in range(N)]

        # Used if there is nothing for one of the rungs
        empty_slice = table.loc[[]]

        # Now for each of our HB brackets, we need to split them into the SH brackets
        hb_brackets: list[list[Sync]] = []

        offsets = np.cumsum([0, *bottom_rung_sizes])
        for hb_ids in hb_id_slices:
            # Split the ids into each of the respective brackets, e.g. [81, 27, 9, ...]
            ids_for_each_bracket = [hb_ids[s:e] for s, e in pairwise(offsets)]

            # Select the data for each of the configs allocated to these sh_brackets
            data_for_each_bracket = [table.loc[_ids] for _ids in ids_for_each_bracket]

            # Create the bracket
            sh_brackets: list[Sync] = []
            for data_for_bracket, layout in zip(
                data_for_each_bracket,
                bracket_layouts,
                strict=True,
            ):
                rung_data = dict(iter(data_for_bracket.groupby(level="rung", sort=False)))
                bracket = Sync(
                    rungs=[
                        Rung(
                            value=rung,
                            capacity=capacity,
                            table=rung_data.get(rung, empty_slice),
                        )
                        for rung, capacity in layout.items()
                    ],
                    is_multi_objective=is_multi_objective,
                    mo_selector=mo_selector,
                )
                sh_brackets.append(bracket)

            hb_brackets.append(sh_brackets)

        return [cls(sh_brackets=sh_brackets) for sh_brackets in hb_brackets]

    def next(self) -> BracketAction:
        # We check what each SH bracket wants to do
        statuses = [sh_bracket.next() for sh_bracket in self.sh_brackets]

        # We define a priority function to sort and decide what to return:
        #
        # 1. "promote"/"new": tie break by rung value
        #   1.1 Tie break by rung value if needed
        #   1.2 Further tie break by index (bracket with lowest rung goes first)
        #         (1.2 is handled implicitly by the sorted order of the brackets
        # 2. "pending": If there are no promotions or new samples, we say HB is pending
        # 3. "done": If everything is done, then we are done.
        def priority(x: BracketAction) -> tuple[int, int]:
            match x:
                case PromoteAction(new_rung=new_rung):
                    return 0, new_rung
                case SampleAction(sample_at_rung):
                    return 1, sample_at_rung
                case "pending":
                    return 2, 0
                case "done":
                    return 3, 0
                case _:
                    raise RuntimeError("This is a bug!")

        sorted_priorities = sorted(statuses, key=priority)  # type: ignore
        return sorted_priorities[0]


@dataclass
class AsyncHyperband:
    asha_brackets: list[Async]
    """A list of ASHA brackets, ordered from lowest to highest according to the lowest
    rung value in each bracket."""

    eta: int
    """The eta parameter used for deciding when to promote."""

    _min_rung: int = field(init=False, repr=False)
    _max_rung: int = field(init=False, repr=False)

    is_multi_objective: bool = field(default=False)
    """Whether the BracketOptimizer is multi-objective or not."""

    mo_selector: Literal["nsga2", "epsnet"] = field(default="epsnet")
    """The selector to use for multi-objective optimization."""

    def __post_init__(self) -> None:
        if not self.asha_brackets:
            raise ValueError("HyperbandBrackets must have at least one ASHA bracket")

        # Sort the brackets by those which contain the lowest rung values first
        self.asha_brackets = sorted(
            self.asha_brackets, key=lambda bracket: bracket.rungs[0].value
        )
        self._min_rung = min(bracket.rungs[0].value for bracket in self.asha_brackets)
        self._max_rung = max(bracket.rungs[-1].value for bracket in self.asha_brackets)

    @classmethod
    def create(
        cls,
        table: pd.DataFrame,
        *,
        bracket_rungs: list[list[int]],
        eta: int,
        is_multi_objective: bool = False,
        mo_selector: Literal["nsga2", "epsnet"] = "epsnet",
    ) -> AsyncHyperband:
        """Create an AsyncHyperbandBrackets from the table.

        The table should have a multi-index of (id, rung) where rung is the
        fidelity level of the configuration.
        ```
        # Unrealistic example showing the format of the table
        (id, rung) -> config, perf
        --------------------------
        0     0   |  {"hp": 0, ...}, 0.1
              1   |  {"hp": 0, ...}, 0.2
        1     0   |  {"hp": 1, ...}, 0.1
        2     1   |  {"hp": 2, ...}, 0.3
              2   |  {"hp": 2, ...}, 0.4
        3     2   |  {"hp": 3, ...}, 0.4
        ```

        Args:
            table: The table of configurations to split into brackets.
            bracket_rungs: A list of rungs for each bracket. Each element of the list
                is a list for that given bracket.

        Returns:
            The AsyncHyperbandBrackets which have each subselected the table with the
            corresponding rung sizes.
        """
        return AsyncHyperband(
            asha_brackets=[
                Async.create(
                    table=table,
                    rungs=layout,
                    eta=eta,
                    is_multi_objective=is_multi_objective,
                    mo_selector=mo_selector,
                )
                for layout in bracket_rungs
            ],
            eta=eta,
        )

    def next(self) -> BracketAction:
        # Each ASHA bracket always has an action, sample which to take
        bracket_ix = async_hb_sample_bracket_to_run(self._max_rung, self.eta)
        bracket = self.asha_brackets[bracket_ix]
        return bracket.next()


Bracket: TypeAlias = Sync | Async | Hyperband | AsyncHyperband
