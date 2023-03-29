from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import scipy

from ...search_spaces import CategoricalParameter, FloatParameter, IntegerParameter
from ...search_spaces.search_space import SearchSpace


def update_fidelity(config, fidelity):
    config.fidelity.value = fidelity
    return config


def local_mutation(
    config: SearchSpace, std: float = 0.25, mutation_rate: float = 0.5, patience: int = 50
) -> SearchSpace:
    """Performs a local search by mutating randomly chosen hyperparameters."""
    for _ in range(patience):
        new_config = deepcopy(config)
        config_hp_names = list(new_config.keys())
        for hp_name in config_hp_names:
            hp = new_config.get(hp_name)
            if hp.is_fidelity or np.random.uniform() > mutation_rate:
                continue
            kwargs = {"mutation_strategy": "local_search"}
            if isinstance(hp, CategoricalParameter):
                confidences = {hp.value: len(hp.choices)}
                kwargs["confidences"] = confidences
            elif isinstance(hp, IntegerParameter) or isinstance(hp, FloatParameter):
                kwargs["std"] = std
            _val = hp.mutate(**kwargs).value
            hp.value = _val
        if not config.is_equal_value(new_config, include_fidelity=False):
            # if the new config doesn't differ from the original config then regenerate
            break

    return new_config


def custom_crossover(
    config1: SearchSpace,
    config2: SearchSpace,
    crossover_prob: float = 0.5,
    patience: int = 50,
) -> SearchSpace:
    """Performs a crossover of config2 into config1.

    Returns a configuration where each HP in config1 has `crossover_prob`% chance of
    getting config2's value of the corresponding HP. By default, crossover rate is 50%.
    """
    for _ in range(patience):
        child_config = deepcopy(config1)
        for key, hyperparameter in config1.items():
            if not hyperparameter.is_fidelity and np.random.random() < crossover_prob:
                # crossing over config2 values into config1
                child_config[key].value = config2[key].value
        if not child_config.is_equal_value(config1):
            # breaks and returns only if the generated config is not the same
            # thus requiring at least 1 HP to be crossed-over
            break
    else:
        # fail safe check to handle edge cases where config1=config2 or
        # config1 extremely local to config2 such that crossover fails to
        # generate new config in a discrete (sub-)space
        child_config = config1.sample(
            patience=patience, user_priors=False, ignore_fidelity=True
        )
    return child_config


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


def calc_total_resources_spent(observed_configs: pd.DataFrame, rung_map: dict) -> float:
    # collects a list of fidelities/rungs reached by configurations that are not pending
    rungs_used = [
        observed_configs.at[i, "rung"]
        for i in range(len(observed_configs))
        if not np.isnan(observed_configs.at[i, "perf"])
    ]
    total_resources = sum(rung_map[r] for r in rungs_used)
    return total_resources


# def get_prior_weight_for_decay(
#     resources_used: float, eta: int, min_budget, max_budget
# ) -> float:
#     nrungs = np.floor(np.log(max_budget / min_budget) / np.log(eta)).astype(int) + 1
#     unit_HB_resources = nrungs * eta * max_budget
#     idx = resources_used // unit_HB_resources
#     start_weight = 1 / eta**idx
#     end_weight = start_weight / eta
#     _resources = resources_used / unit_HB_resources - idx
#
#     # equation for line in the idx-th HB bracket in terms of resource usage
#     y = (end_weight - start_weight) * _resources + start_weight
#
#     return y


def get_prior_weight_for_decay(
    resources_used: float, eta: int, min_budget, max_budget
) -> float:
    """Creates a step function schedule for the prior weight decay.

    The prior weight ratio is decayed every time the total resources used is
    equivalent to the cost of one successive halving bracket within the HB schedule.
    This is approximately eta \times max_budget resources for one evaluation.
    """
    # decay factor for the prior
    decay = 2
    unit_resources = eta * max_budget
    idx = resources_used // unit_resources
    weight = 1 / decay**idx
    return weight
