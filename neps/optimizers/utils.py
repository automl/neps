from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace


def map_real_hyperparameters_from_tabular_ids(
    x: pd.Series, pipeline_space: SearchSpace
) -> pd.Series:
    """Maps the tabular IDs to the actual HPs from the pipeline space.

    Args:
        x (pd.Series): A pandas series with the tabular IDs.
            TODO: Mention expected format of the series.
        pipeline_space (SearchSpace): The pipeline space.

    Returns:
        pd.Series: A pandas series with the actual HPs.
            TODO: Mention expected format of the series.
    """
    if len(x) == 0:
        return x
    # copying hyperparameter configs based on IDs
    _x = pd.Series(
        [
            pipeline_space.custom_grid_table[x.loc[idx]["id"].value]
            for idx in x.index.values
        ],
        index=x.index,
    )
    # setting the passed fidelities for the corresponding IDs
    for idx in _x.index.values:
        _x.loc[idx].fidelity.value = x.loc[idx].fidelity.value
    return _x
