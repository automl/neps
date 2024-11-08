from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd


class MFObservedData:
    """(Under development).

    This module is used to unify the data access across different Multi-Fidelity
    optimizers. It stores column names and index names. Possible optimizations
    and extensions of the observed data should be handled by this class.

    So far this is just a draft class containing the DataFrame and some properties.
    """

    default_config_idx = "config_id"
    default_budget_idx = "budget_id"
    default_config_col = "config"
    default_perf_col = "perf"
    default_lc_col = "learning_curves"
    # TODO: deepcopy all the mutable outputs from the dataframe

    def __init__(
        self,
        columns: list[str] | None = None,
        index_names: list[str] | None = None,
    ):
        if columns is None:
            columns = [self.default_config_col, self.default_perf_col]
        if index_names is None:
            index_names = [self.default_config_idx, self.default_budget_idx]

        self.config_col = columns[0]
        self.perf_col = columns[1]

        if len(columns) > 2:
            self.lc_col_name = columns[2]
        else:
            self.lc_col_name = self.default_lc_col

        if len(index_names) == 1:
            index_names += ["budget_id"]

        self.config_idx = index_names[0]
        self.budget_idx = index_names[1]
        self.index_names = index_names

        index = pd.MultiIndex.from_tuples([], names=index_names)

        self.df = pd.DataFrame([], columns=columns, index=index)

    @property
    def pending_condition(self) -> pd.Series:
        return self.df[self.perf_col].isna()

    @property
    def error_condition(self) -> pd.Series:
        return self.df[self.perf_col] == "error"

    @property
    def seen_config_ids(self) -> list:
        return self.df.index.levels[0].to_list()

    @property
    def seen_budget_levels(self) -> list:
        # Considers pending and error budgets as seen
        return self.df.index.levels[1].to_list()

    @property
    def pending_runs_index(self) -> pd.Index | pd.MultiIndex:
        return self.df.loc[self.pending_condition].index

    @property
    def completed_runs(self) -> pd.DataFrame:
        return self.df[~(self.pending_condition | self.error_condition)]

    @property
    def completed_runs_index(self) -> pd.Index | pd.MultiIndex:
        return self.completed_runs.index

    def next_config_id(self) -> int:
        if len(self.seen_config_ids):
            return max(self.seen_config_ids) + 1
        return 0

    def add_data(
        self,
        data: list[Any] | list[list[Any]],
        index: tuple[int, ...] | Sequence[tuple[int, ...]] | Sequence[int] | int,
        *,
        error: bool = False,
    ) -> None:
        """Add data only if none of the indices are already existing in the DataFrame."""
        # TODO: If index is only config_id extend it
        if not isinstance(index, list):
            index_list = [index]
            data_list = [data]
        else:
            index_list = index
            data_list = data

        if not self.df.index.isin(index_list).any():
            index = pd.MultiIndex.from_tuples(index_list, names=self.index_names)
            _df = pd.DataFrame(data_list, columns=self.df.columns, index=index)
            self.df = _df.copy() if self.df.empty else pd.concat((self.df, _df))
        elif error:
            raise ValueError(
                f"Data with at least one of the given indices already "
                f"exists: {self.df[self.df.index.isin(index_list)]}\n"
                f"Given indices: {index_list}"
            )

    def update_data(
        self,
        data_dict: dict[str, list[Any]],
        index: tuple[int, ...] | Sequence[tuple[int, ...]] | Sequence[int] | int,
        *,
        error: bool = False,
    ) -> None:
        """Update data if all the indices already exist in the DataFrame."""
        index_list = [index] if not isinstance(index, list) else index
        if self.df.index.isin(index_list).sum() == len(index_list):
            column_names, data = zip(*data_dict.items(), strict=False)
            data = list(zip(*data, strict=False))
            self.df.loc[index_list, list(column_names)] = data

        elif error:
            raise ValueError(
                f"Data with at least one of the given indices doesn't "
                f"exist.\n Existing indices: {self.df.index}\n"
                f"Given indices: {index_list}"
            )

    def get_learning_curves(self) -> pd.DataFrame:
        return self.df.pivot_table(
            index=self.df.index.names[0],
            columns=self.df.index.names[1],
            values=self.perf_col,
        )

    def all_configs_list(self) -> list[Any]:
        return self.df.loc[:, self.config_col].sort_index().values.tolist()

    def get_best_learning_curve_id(self, *, maximize: bool = False) -> int:
        """Returns a single configuration id of the best observed performance.

        Note: this will always return the single best lowest ID
              if two configurations has the same performance
        """
        learning_curves = self.get_learning_curves()
        if maximize:
            return learning_curves.max(axis=1).idxmax()
        return learning_curves.min(axis=1).idxmin()

    def get_best_seen_performance(self, *, maximize: bool = False) -> float:
        learning_curves = self.get_learning_curves()
        if maximize:
            return learning_curves.max(axis=1).max()
        return learning_curves.min(axis=1).min()

    def add_budget_column(self) -> pd.DataFrame:
        combined_df = self.df.reset_index(level=1)
        return combined_df.set_index(keys=[self.budget_idx], drop=False, append=True)

    def reduce_to_max_seen_budgets(self) -> pd.DataFrame:
        self.df = self.df.sort_index()
        combined_df = self.add_budget_column()
        return combined_df.groupby(level=0).last()

    def get_partial_configs_at_max_seen(self) -> pd.Series:
        return self.reduce_to_max_seen_budgets()[self.config_col]

    def extract_learning_curve(
        self, config_id: int, budget_id: int | None = None
    ) -> list[float]:
        if budget_id is None:
            # budget_id only None when predicting
            # extract full observed learning curve for prediction pipeline
            budget_id = (
                max(self.df.loc[config_id].index.get_level_values("budget_id").values) + 1
            )

        # For the first epoch we have no learning curve available
        if budget_id == 0:
            return []
        # reduce budget_id to discount the current validation objective_to_minimize
        # both during training and prediction phase
        budget_id = max(0, budget_id - 1)
        if self.lc_col_name in self.df.columns:
            lc = self.df.loc[(config_id, budget_id), self.lc_col_name]
        else:
            lcs = self.get_learning_curves()
            lc = lcs.loc[config_id, :budget_id].values.flatten().tolist()
        return deepcopy(lc)

    def get_best_performance_per_config(self, *, maximize: bool = False) -> pd.Series:
        """Returns the best score recorded per config across fidelities seen."""
        op = np.max if maximize else np.min
        return (
            self.df.sort_values(
                "budget_id", ascending=False
            )  # sorts with largest budget first
            .groupby("config_id")  # retains only config_id
            .first()  # retrieves the largest budget seen for each config_id
            .learning_curves.apply(  # extracts all values seen till largest budget
                op
            )  # finds the minimum over per-config learning curve
        )

    def get_max_observed_fidelity_level_per_config(self) -> pd.Series:
        """Returns the highest fidelity level recorded per config seen."""
        max_z_observed = {
            _id: self.df.loc[_id, :].index.sort_values()[-1]
            for _id in self.df.index.get_level_values("config_id").sort_values()
        }
        return pd.Series(max_z_observed)


if __name__ == "__main__":
    # TODO: Either delete these or convert them to tests (karibbov)
    """
    Here are a few examples of how to manage data with this class:
    """
    data = MFObservedData(["config", "perf"], index_names=["config_id", "budget_id"])

    # When adding multiple indices data should be list of rows(lists) and the
    # index should be list of tuples
    data.add_data(
        [["conf1", 0.5], ["conf2", 0.7], ["conf1", 0.6], ["conf2", 0.4]],
        index=[(0, 0), (1, 1), (0, 3), (1, 0)],
    )
    data.add_data(
        [["conf1", 0.5], ["conf2", 0.10], ["conf1", 0.11]],
        index=[(0, 2), (1, 2), (0, 1)],
    )

    # When updating multiple indices at a time both the values in the data dictionary
    # and the indices should be lists
    data.update_data({"perf": [1.8, 1.5]}, index=[(1, 1), (0, 0)])

    data = MFObservedData(["config", "perf"], index_names=["config_id", "budget_id"])

    # when adding a single row second level list is not necessary
    data.add_data(["conf1", 0.5], index=(0, 0))

    data.update_data({"perf": [1.8], "budget_col": [5]}, index=(0, 0))
