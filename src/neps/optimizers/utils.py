import pandas as pd

from ..search_spaces.search_space import SearchSpace


# def map_real_hyperparameters_from_tabular_ids(
#     ids: pd.Series, pipeline_space: SearchSpace
# ) -> pd.Series:
#         return x
        

def map_real_hyperparameters_from_tabular_ids(
    x: pd.Series, pipeline_space: SearchSpace
) -> pd.Series:
    """ Maps the tabular IDs to the actual HPs from the pipeline space.
    
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
    # extract fid name
    _x = x.iloc[0].hp_values()
    _x.pop("id")
    fid_name = list(_x.keys())[0]
    for i in x.index.values:
        # extracting actual HPs from the tabular space
        _config = pipeline_space.custom_grid_table.loc[x.loc[i]["id"].value].to_dict()
        # updating fidelities as per the candidate set passed
        _config.update({fid_name: x.loc[i][fid_name].value})
        # placeholder config from the raw tabular space
        config = pipeline_space.raw_tabular_space.sample(
            patience=100, 
            user_priors=True, 
            ignore_fidelity=True  # True allows fidelity to appear in the sample
        )
        # copying values from table to placeholder config of type SearchSpace
        config.load_from(_config)
        # replacing the ID in the candidate set with the actual HPs of the config
        x.loc[i] = config
    return x
