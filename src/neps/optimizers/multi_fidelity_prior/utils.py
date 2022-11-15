from copy import deepcopy
from typing import Dict, Union

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

    # TODO: can we consider the number of choices per dimension
    # computing hamming distance over the categorical subspace
    d_cat = scipy.spatial.distance.hamming(
        config1["categorical"] + [0], config2["categorical"] + [0]
    )

    distance = d_cont + d_cat
    return distance


def compute_scores(
    config: SearchSpace, prior: SearchSpace, inc: SearchSpace
) -> Union[float, float]:
    """Scores the config by a Gaussian around the prior and the incumbent."""
    _prior = deepcopy(prior)
    _prior.set_hyperparameters_from_dict(config.hp_values(), defaults=False)
    # compute the score of config if it was sampled from the prior (as the default)
    prior_score = _prior.compute_prior()

    _inc = deepcopy(inc)
    # setting the default to be the incumbent
    _inc.set_defaults_to_current_values()
    _inc.set_hyperparameters_from_dict(config.hp_values(), defaults=False)
    # compute the score of config if it was sampled from the inc (as the default)
    inc_score = _inc.compute_prior()

    return prior_score, inc_score


class DynamicWeights:
    def prior_inc_probability_ratio(
        self, rung_history: Dict[str, Dict], prior: SearchSpace, inc: SearchSpace
    ) -> Union[float, float]:
        # uses the base rung size to determine the maximum size of top config list
        max_rung_size = self.config_map[self.min_rung]
        if inc is None:
            return 1, 0
        elif len(rung_history["config"]) < self.eta:
            inc_score = 1
            prior_score = self.eta * inc_score
        else:
            # subset top configs
            top_len = min(len(rung_history["config"]) // self.eta, max_rung_size)
            config_idxs = np.argsort(rung_history["perf"])[:top_len]
            top_configs = np.array(rung_history["config"])[config_idxs]

            # calculating scores for each config
            top_config_scores = np.array(
                [
                    compute_scores(
                        self.observed_configs.loc[config_id].config, prior, inc
                    )
                    for config_id in top_configs
                ]
            )
            # adding weights as per rank of config
            w = np.flip(np.arange(1, top_config_scores.shape[0] + 1)).reshape(-1, 1)
            w_top_config_scores = top_config_scores * w  # / w.sum()
            prior_score, inc_score = np.sum(w_top_config_scores, axis=0)

        # calculating probabilities
        # normalizer = np.exp(prior_score) + np.exp(inc_score)
        # prior_prob = np.exp(prior_score) / normalizer
        # inc_prob = np.exp(inc_score) / normalizer

        _inc_prob = 0.333
        delta = (prior_score - inc_score) / max(prior_score, inc_score)
        # print(f"prior_score: {prior_score:.3f}; inc_score: {inc_score:.3f}")
        # print(f"delta: {delta:.3f}")
        _inc_prob -= _inc_prob * delta
        _inc_prob = np.clip(_inc_prob, a_min=0, a_max=0.5)
        # _inc_prob = np.clip(_inc_prob - delta, a_min=0, a_max=0.5)
        # print(f"new inc prob: {_inc_prob}")
        inc_prob = _inc_prob
        prior_prob = 1 - inc_prob

        return prior_prob, inc_prob
