from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from typing import Any

from neps.search_spaces import (
    CategoricalParameter,
    ConstantParameter,
    GraphParameter,
    SearchSpace,
    NumericalParameter,
)
from neps.search_spaces.config import Config
import neps.search_spaces.neighborhoods as neighborhoods


def update_fidelity(config, fidelity):
    config.fidelity.set_value(fidelity)
    return config


# TODO(eddiebergman): Previously this just ignored graphs,
# now it will likely raise if it encounters one...
def local_mutation(
    config: Config,
    space: SearchSpace,
    *,
    seed: np.random.Generator,
    std: float = 0.25,
    mutation_rate: float = 0.5,
    patience: int = 50,
    mutate_categoricals: bool = True,
    mutate_graphs: bool = True,
) -> Config:
    """Performs a local search by mutating randomly chosen hyperparameters."""
    for _ in range(patience):
        new_config: dict[str, Any] = {}

        for hp_name, hp in space.hyperparameters.items():
            value = config.values[hp_name]
            if np.random.uniform() > mutation_rate:
                new_config[hp_name] = value

            elif isinstance(hp, CategoricalParameter):
                if mutate_categoricals:
                    new_config[hp_name] = neighborhoods.unorded_finite_neighbors(
                        pivot=config.values[hp_name],
                        domain=hp.domain,
                        n=1,
                        seed=seed,
                    )
                else:
                    new_config[hp_name] = value

            elif isinstance(hp, GraphParameter):
                if mutate_graphs:
                    new_config[hp_name] = hp.mutate(mutation_strategy="bananas")
                else:
                    new_config[hp_name] = value

            elif isinstance(hp, NumericalParameter):
                new_config[hp_name] = neighborhoods.unique_neighborhood(
                    pivot=value,
                    domain=hp.domain,
                    n=1,
                    seed=seed,
                    std=std,
                )
            elif isinstance(hp, ConstantParameter):
                new_config[hp_name] = value
            else:
                raise NotImplementedError(f"Unknown hp type for {hp_name}: {type(hp)}")

        # if the new config doesn't differ from the original config then regenerate
        if new_config != config.values:
            return Config(values=new_config, fidelity=config.fidelity)

    return config.clone()


def custom_crossover(
    config1: Config,
    config2: Config,
    space: SearchSpace,
    crossover_prob: float = 0.5,
    patience: int = 50,
) -> Config | None:
    """Performs a crossover of config2 into config1.

    Returns a configuration where each HP in config1 has `crossover_prob`% chance of
    getting config2's value of the corresponding HP. By default, crossover rate is 50%.
    """
    new_values = {}
    for _ in range(patience):
        for key in space.hyperparameters:
            if np.random.random() < crossover_prob:
                new_values[key] = config2.values[key]
            else:
                new_values[key] = config1.values[key]

        if new_values == config1.values:
            return config1.clone(values=new_values)

    return None


def compute_config_dist(config1: Config, config2: Config) -> float:
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
    config: SearchSpace,
    prior: SearchSpace,
    inc: SearchSpace,
) -> tuple[float, float]:
    """Scores the config by a Gaussian around the prior and the incumbent."""
    _prior = prior.clone()
    _prior.set_hyperparameters_from_dict(config.deprecated_hp_values(), defaults=False)
    # compute the score of config if it was sampled from the prior (as the default)
    prior_score = _prior.compute_prior()

    _inc = inc.clone()
    # setting the default to be the incumbent
    _inc.set_defaults_to_current_values()
    _inc.set_hyperparameters_from_dict(config.deprecated_hp_values(), defaults=False)
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
