import numpy as np
import scipy

from ...search_spaces.search_space import SearchSpace


def compute_config_dist(config1: SearchSpace, config2: SearchSpace) -> float:
    """Computes distance between two configurations.

    Divides the search space into continuous and categorical subspaces.
    Normalizes all the continuous values while gives numerical encoding to categories.
    Distance returned is the sum of the Euclidean distance of the continous subspace and
    the Hamming distance of the categorical subspace.
    """
    config1 = config1.get_normalized_hp_categories(ignore_fidelity=True)
    config2 = config2.get_normalized_hp_categories(ignore_fidelity=True)

    # adding a dim with 0 to all subspaces in case the search space is not mixed type

    # computing euclidean distance over the continuous subspace
    diff = np.array(config1["continuous"] + [0]) - np.array(config2["continuous"] + [0])
    d_cont = np.linalg.norm(diff, ord=2)

    # computing hamming distance over the categorical subspace
    d_cat = scipy.spatial.distance.hamming(
        config1["categorical"] + [0], config2["categorical"] + [0]
    )

    distance = d_cont + d_cat
    return distance
