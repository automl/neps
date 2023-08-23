from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class MFObservedData:
    """
    (Under development)

    This module is used to unify the data access across different Multi-Fidelity
    optimizers. It stores column names and index names. Possible optimizations
    and extensions of the observed data should be handled by this class.

    So far this is just a draft class containing the DataFrame and some properties.
    """

    def __init__(
        self,
        columns: Union[List[str], None] = None,
        index_names: Union[List[str], None] = None,
        config_column_name: str = "config",
        perf_column_name: str = "perf",
        budget_level_column_name: str = "budget_id",
    ):
        if columns is None:
            columns = ["config", "perf", "budget_id"]
        if index_names is None:
            index_names = ["config_id", "budget_id"]

        self.config_col = config_column_name
        self.perf_col = perf_column_name
        self.budget_col = budget_level_column_name

        if len(index_names) > 1:
            index = pd.MultiIndex.from_tuples([], names=index_names)
        else:
            index = pd.Index([], name=index_names[0], dtype=np.int64)

        self.df = pd.DataFrame([], columns=columns, index=index)

    @property
    def pending_condition(self):
        return self.df[self.perf_col].isnull()

    @property
    def error_condition(self):
        return self.df[self.perf_col] == "error"

    @property
    def completed_runs(self):
        return self.df[~(self.pending_condition | self.error_condition)]

    def add_data(
        self,
        data: Union[List[Any], List[List[Any]]],
        index: Union[Tuple[int, ...], Sequence[Tuple[int, ...]], Sequence[int], int],
        error: bool = False,
    ):
        """
        Add data only if none of the indices are already existing in the DataFrame
        """
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
        data_dict: Dict[str, List[Any]],
        index: Union[Tuple[int, ...], Sequence[Tuple[int, ...]], Sequence[int], int],
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


if __name__ == "__main__":
    # TODO: Either delete these or convert them to tests (karibbov)
    """
    Here are a few examples of how to manage data with this class:
    """
    # Multi Index case:
    data = MFObservedData(
        ["config", "perf", "budget_id"], index_names=["config_id", "budget_id"]
    )

    # When adding multiple indices data should be list of rows(lists) and the index should be list of tuples
    data.add_data(
        [["conf1", 0.5, 0], ["conf2", 0.8, 1], ["conf1", 0.9, 1], ["conf2", 0.4, 0]],
        index=[(0, 0), (1, 1), (0, 1), (1, 0)],
    )
    print(data.df)

    # When updating multiple indices at a time both the values in the data dictionary and the indices should be lists
    data.update_data({"perf": [1.8, 1.5], "budget_id": [5, 6]}, index=[(1, 1), (0, 0)])
    print(data.df)

    data = MFObservedData(
        ["config", "perf", "budget_id"], index_names=["config_id", "budget_id"]
    )

    # when adding a single row second level list is not necessary
    data.add_data(["conf1", 0.5, 0], index=(0, 0))
    print(data.df)

    data.update_data({"perf": [1.8], "budget_id": [5]}, index=(0, 0))
    print(data.df)

    # Single index case:
    data = MFObservedData(["config", "perf", "budget_id"], index_names=["config_id"])

    data.add_data(
        [["conf1", 0.5, 0], ["conf2", 0.8, 1], ["conf1", 0.9, 1], ["conf2", 0.4, 0]],
        index=[(0), (1), (2), (3)],
    )
    print(data.df)

    data.update_data({"perf": [1.8, 1.5], "budget_id": [5, 6]}, index=[(1), (0)])
    print(data.df)

    data = MFObservedData(["config", "perf", "budget_id"], index_names=["config_id"])

    data.add_data(["conf1", 0.5, 0], index=0)
    print(data.df)

    data.update_data({"perf": [1.8], "budget_id": [5]}, index=0)
    print(data.df)
