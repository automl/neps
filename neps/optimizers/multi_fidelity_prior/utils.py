from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from neps.sampling.priors import Prior
from neps.search_spaces import (
    CategoricalParameter,
    ConstantParameter,
    GraphParameter,
    NumericalParameter,
    Parameter,
    SearchSpace,
)
from neps.search_spaces.encoding import ConfigEncoder

if TYPE_CHECKING:
    import pandas as pd


def update_fidelity(config: SearchSpace, fidelity: int | float) -> SearchSpace:
    assert config.fidelity is not None
    config.fidelity.set_value(fidelity)
    return config


# TODO(eddiebergman): Previously this just ignored graphs,
# now it will likely raise if it encounters one...
def local_mutation(
    config: SearchSpace,
    std: float = 0.25,
    mutation_rate: float = 0.5,
    patience: int = 50,
    mutate_categoricals: bool = True,
    mutate_graphs: bool = True,
) -> SearchSpace:
    """Performs a local search by mutating randomly chosen hyperparameters."""
    # Used to check uniqueness later.
    _existing = {
        k: v for k, v in config.hp_values().items() if k not in config.fidelities
    }

    for _ in range(patience):
        new_config: dict[str, Parameter] = {}

        for hp_name, hp in config.items():
            if hp.is_fidelity or np.random.uniform() > mutation_rate:
                new_config[hp_name] = hp.clone()

            elif isinstance(hp, CategoricalParameter):
                if mutate_categoricals:
                    new_config[hp_name] = hp.mutate(mutation_strategy="local_search")
                else:
                    new_config[hp_name] = hp.clone()

            elif isinstance(hp, GraphParameter):
                if mutate_graphs:
                    new_config[hp_name] = hp.mutate(mutation_strategy="bananas")
                else:
                    new_config[hp_name] = hp.clone()

            elif isinstance(hp, NumericalParameter):
                new_config[hp_name] = hp.mutate(
                    mutation_strategy="local_search",
                    std=std,
                )
            elif isinstance(hp, ConstantParameter):
                new_config[hp_name] = hp.clone()

            else:
                raise NotImplementedError(f"Unknown hp type for {hp_name}: {type(hp)}")

        # if the new config doesn't differ from the original config then regenerate
        _new = {k: v for k, v in new_config.items() if k not in config.fidelities}
        if _existing != _new:
            return SearchSpace(**new_config)

    return config.clone()


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
    _existing = config1.hp_values()

    for _ in range(patience):
        child_config = {}
        for key, hyperparameter in config1.items():
            if not hyperparameter.is_fidelity and np.random.random() < crossover_prob:
                child_config[key] = config2[key].value
            else:
                child_config[key] = hyperparameter.value

        if _existing != child_config:
            return SearchSpace(**child_config)

    # fail safe check to handle edge cases where config1=config2 or
    # config1 extremely local to config2 such that crossover fails to
    # generate new config in a discrete (sub-)space
    return config1.sample(
        patience=patience,
        user_priors=False,
        ignore_fidelity=True,
    )


def compute_config_dist(config1: SearchSpace, config2: SearchSpace) -> float:
    """Computes distance between two configurations.

    Divides the search space into continuous and categorical subspaces.
    Normalizes all the continuous values while gives numerical encoding to categories.
    Distance returned is the sum of the Euclidean distance of the continous subspace and
    the Hamming distance of the categorical subspace.
    """
    encoder = ConfigEncoder.default({**config1.numerical, **config1.categoricals})
    configs = encoder.encode([config1.hp_values(), config2.hp_values()])
    dist = encoder.pdist(configs, square_form=False)
    return float(dist.item())


def compute_scores(
    config: SearchSpace,
    prior: SearchSpace,
    inc: SearchSpace,
) -> tuple[float, float]:
    """Scores the config by a Gaussian around the prior and the incumbent."""
    _prior = prior.clone()
    _prior.set_hyperparameters_from_dict(config.hp_values(), defaults=False)
    # compute the score of config if it was sampled from the prior (as the default)
    prior_score = _prior.compute_prior()

    _inc = inc.clone()
    # setting the default to be the incumbent
    _inc.set_defaults_to_current_values()
    _inc.set_hyperparameters_from_dict(config.hp_values(), defaults=False)
    # compute the score of config if it was sampled from the inc (as the default)
    inc_score = _inc.compute_prior()

    return prior_score, inc_score


def get_prior_weight_for_decay(
    resources_used: float, eta: int, min_budget: int | float, max_budget: int | float
) -> float:
    r"""Creates a step function schedule for the prior weight decay.

    The prior weight ratio is decayed every time the total resources used is
    equivalent to the cost of one successive halving bracket within the HB schedule.
    This is approximately eta \times max_budget resources for one evaluation.
    """
    # decay factor for the prior
    decay = 2
    unit_resources = eta * max_budget
    idx = resources_used // unit_resources
    return 1 / decay**idx
