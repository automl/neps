import numpy as np


def init_random_uniform(lower, upper, n_points, n_dims, rng=None):

    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    return np.array([rng.uniform(lower, upper, n_dims) for _ in range(n_points)]).tolist()
