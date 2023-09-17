from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ...search_spaces.search_space import SearchSpace


def continuous_to_tabular(
    config: SearchSpace, categorical_space: SearchSpace
) -> SearchSpace:
    """
    Convert the continuous parameters in the config into categorical ones based on
    the categorical_space provided
    """
    result = config.copy()
    for hp_name, _ in config.items():
        if hp_name in categorical_space.keys():
            choices = np.array(categorical_space[hp_name].choices)
            diffs = choices - config[hp_name].value
            # NOTE: in case of a tie the first value in the choices array will be returned
            closest = choices[np.abs(diffs).argmin()]
            result[hp_name].value = closest

    return result


class MFObservedData:
    """
    (Under development)

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

        index = pd.MultiIndex.from_tuples([], names=index_names)

        self.df = pd.DataFrame([], columns=columns, index=index)

    @property
    def pending_condition(self):
        return self.df[self.perf_col].isnull()

    @property
    def error_condition(self):
        return self.df[self.perf_col] == "error"

    @property
    def seen_config_ids(self) -> list:
        return self.df.index.levels[0].to_list()

    @property
    def completed_runs(self):
        return self.df[~(self.pending_condition | self.error_condition)]

    def next_config_id(self) -> int:
        if len(self.seen_config_ids):
            return max(self.seen_config_ids) + 1
        else:
            return 0

    def add_data(
        self,
        data: list[Any] | list[list[Any]],
        index: tuple[int, ...] | Sequence[tuple[int, ...]] | Sequence[int] | int,
        error: bool = False,
    ):
        """
        Add data only if none of the indices are already existing in the DataFrame
        """
        # TODO: If index is only config_id extend it
        if not isinstance(index, list):
            index_list = [index]
            data_list = [data]
        else:
            index_list = index
            data_list = data

        if not self.df.index.isin(index_list).any():
            _df = pd.DataFrame(data_list, columns=self.df.columns, index=index_list)
            self.df = pd.concat((self.df, _df))
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
        error: bool = False,
    ):
        """
        Update data if all the indices already exist in the DataFrame
        """
        if not isinstance(index, list):
            index_list = [index]
        else:
            index_list = index
        if self.df.index.isin(index_list).sum() == len(index_list):
            column_names, data = zip(*data_dict.items())
            data = list(zip(*data))
            self.df.loc[index_list, list(column_names)] = data

        elif error:
            raise ValueError(
                f"Data with at least one of the given indices doesn't "
                f"exist.\n Existing indices: {self.df.index}\n"
                f"Given indices: {index_list}"
            )

    def get_learning_curves(self):
        return self.df.pivot_table(
            index=self.df.index.names[0],
            columns=self.df.index.names[1],
            values=self.perf_col,
        )

    def all_configs_list(self) -> list[Any]:
        return self.df.loc[:, self.config_col].values.tolist()

    def get_incumbents_for_budgets(self, maximize: bool = False):
        """
        Returns a series object with the best partial configuration for each budget id

        Note: this will always map the best lowest ID if two configurations
              has the same performance at the same fidelity
        """
        learning_curves = self.get_learning_curves()
        if maximize:
            config_ids = learning_curves.idxmax(axis=0)
        else:
            config_ids = learning_curves.idxmin(axis=0)

        indices = list(zip(config_ids.values.tolist(), config_ids.index.to_list()))
        partial_configs = self.df.loc[indices, self.config_col].to_list()
        return pd.Series(partial_configs, index=config_ids.index, name=self.config_col)

    def get_best_performance_for_each_budget(self, maximize: bool = False):
        """
        Returns a series object with the best partial configuration for each budget id

        Note: this will always map the best lowest ID if two configurations
              has the same performance at the same fidelity
        """
        learning_curves = self.get_learning_curves()
        if maximize:
            performance = learning_curves.max(axis=0)
        else:
            performance = learning_curves.min(axis=0)

        return performance

    def get_best_learning_curve_id(self, maximize: bool = False):
        """
        Returns a single configuration id of the best observed performance

        Note: this will always return the single best lowest ID
              if two configurations has the same performance
        """
        learning_curves = self.get_learning_curves()
        if maximize:
            return learning_curves.max(axis=1).idxmax()
        else:
            return learning_curves.min(axis=1).idxmin()

    def get_best_seen_performance(self, maximize: bool = False):
        learning_curves = self.get_learning_curves()
        if maximize:
            return learning_curves.max(axis=1).max()
        else:
            return learning_curves.min(axis=1).min()

    def add_budget_column(self):
        combined_df = self.df.reset_index(level=1)
        combined_df.set_index(
            keys=[self.budget_idx], drop=False, append=True, inplace=True
        )
        return combined_df

    def reduce_to_max_seen_budgets(self):
        self.df.sort_index(inplace=True)
        combined_df = self.add_budget_column()
        return combined_df.groupby(level=0).last()

    def get_partial_configs_at_max_seen(self):
        return self.reduce_to_max_seen_budgets()[self.config_col]

    def extract_learning_curve(self, config_id: int, budget_id: int) -> list[float]:
        if self.lc_col_name in self.df.columns:
            lc = self.df.loc[(config_id, budget_id), self.lc_col_name]
        else:
            lcs = self.get_learning_curves()
            lc = lcs.loc[config_id, :budget_id].values.flatten().tolist()
        return lc

    def get_training_data_4DyHPO(self, df: pd.DataFrame):
        configs = []
        learning_curves = []
        performance = []
        for idx, row in df.iterrows():
            config_id = idx[0]
            budget_id = idx[1]
            configs.append(row[self.config_col])
            performance.append(row[self.perf_col])
            learning_curves.append(self.extract_learning_curve(config_id, budget_id))
        return configs, learning_curves, performance


if __name__ == "__main__":
    # TODO: Either delete these or convert them to tests (karibbov)
    """
    Here are a few examples of how to manage data with this class:
    """
    data = MFObservedData(["config", "perf"], index_names=["config_id", "budget_id"])

    # When adding multiple indices data should be list of rows(lists) and the index should be list of tuples
    data.add_data(
        [["conf1", 0.5], ["conf2", 0.7], ["conf1", 0.6], ["conf2", 0.4]],
        index=[(0, 0), (1, 1), (0, 3), (1, 0)],
    )
    data.add_data(
        [["conf1", 0.5], ["conf2", 0.10], ["conf1", 0.11]],
        index=[(0, 2), (1, 2), (0, 1)],
    )

    print(data.df)
    print(data.get_learning_curves())
    print(
        "Mapping of budget IDs into best performing configurations at each fidelity:\n",
        data.get_incumbents_for_budgets(),
    )
    print(
        "Best Performance at each budget level:\n",
        data.get_best_performance_for_each_budget(),
    )
    print(
        "Configuration ID of the best observed performance so far: ",
        data.get_best_learning_curve_id(),
    )
    print(data.extract_learning_curve(0, 2))
    # data.df.sort_index(inplace=True)
    print(data.get_partial_configs_at_max_seen())

    # When updating multiple indices at a time both the values in the data dictionary and the indices should be lists
    data.update_data({"perf": [1.8, 1.5]}, index=[(1, 1), (0, 0)])
    print(data.df)

    data = MFObservedData(["config", "perf"], index_names=["config_id", "budget_id"])

    # when adding a single row second level list is not necessary
    data.add_data(["conf1", 0.5], index=(0, 0))
    print(data.df)

    data.update_data({"perf": [1.8], "budget_col": [5]}, index=(0, 0))
    print(data.df)
