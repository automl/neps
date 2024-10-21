from __future__ import annotations

from typing import Any

import numpy as np
import torch

from neps.sampling.priors import Prior
from neps.search_spaces import (
    Categorical,
    Constant,
    GraphParameter,
    Float,
    Integer,
    SearchSpace,
)
from neps.search_spaces.encoding import ConfigEncoder
from neps.search_spaces.functions import sample_one_old, pairwise_dist


def update_fidelity(config: SearchSpace, fidelity: int | float) -> SearchSpace:
    assert config.fidelity is not None
    config.fidelity.set_value(fidelity)
    return config


# TODO(eddiebergman): This would be much faster
# if done in a vectorized manner...
def local_mutation(
    config: SearchSpace,
    std: float = 0.25,
    mutation_rate: float = 0.5,
    patience: int = 50,
) -> SearchSpace:
    """Performs a local search by mutating randomly chosen hyperparameters."""
    # Used to check uniqueness later.
    # TODO: Seeding
    space = config
    parameters_to_keep = {}
    parameters_to_mutate = {}

    for name, parameter in space.hyperparameters.items():
        if (
            parameter.is_fidelity
            or isinstance(parameter, Constant)
            or np.random.uniform() > mutation_rate
        ):
            parameters_to_keep[name] = parameter.value
        else:
            parameters_to_mutate[name] = parameter

    if len(parameters_to_mutate) == 0:
        return space.from_dict(parameters_to_keep)

    new_config: dict[str, Any] = {}

    for hp_name, hp in parameters_to_mutate.items():
        match hp:
            case Categorical():
                assert hp._value_index is not None
                perm: list[int] = torch.randperm(len(hp.choices)).tolist()
                ix = perm[0] if perm[0] != hp._value_index else perm[1]
                new_config[hp_name] = hp.choices[ix]
            case GraphParameter():
                new_config[hp_name] = hp.mutate(mutation_strategy="bananas")
            case Integer() | Float():
                prior = Prior.from_parameters(
                    {hp_name: hp},
                    confidence_values={hp_name: (1 - std)},
                )

                for _ in range(patience):
                    sample = prior.sample(1, to=hp.domain).item()
                    if sample != hp.value:
                        new_config[hp_name] = hp.value
                        break
                else:
                    raise ValueError(
                        f"Exhausted patience trying to mutate parameter '{hp_name}'"
                        f" with value {hp.value}"
                    )
            case _:
                raise NotImplementedError(f"Unknown hp type for {hp_name}: {type(hp)}")

    return space.from_dict(new_config)


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
    _existing = config1._values

    for _ in range(patience):
        child_config = {}
        for key, hyperparameter in config1.items():
            if not hyperparameter.is_fidelity and np.random.random() < crossover_prob:
                child_config[key] = config2[key].value
            else:
                child_config[key] = hyperparameter.value

        if _existing != child_config:
            return config1.from_dict(child_config)

    # fail safe check to handle edge cases where config1=config2 or
    # config1 extremely local to config2 such that crossover fails to
    # generate new config in a discrete (sub-)space
    return sample_one_old(
        config1,
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
    encoder = ConfigEncoder.from_parameters({**config1.numerical, **config1.categoricals})
    configs = encoder.encode([config1._values, config2._values])
    dist = pairwise_dist(configs, encoder, square_form=False)
    return float(dist.item())


def compute_scores(
    config: SearchSpace,
    prior: SearchSpace,
    inc: SearchSpace,
    *,
    include_fidelity: bool = False,
) -> tuple[float, float]:
    """Scores the config by a Gaussian around the prior and the incumbent."""
    # TODO: This could lifted up and just done in the class itself
    # in a vectorized form.
    encoder = ConfigEncoder.from_space(config, include_fidelity=include_fidelity)
    encoded_config = encoder.encode([config._values])

    prior_dist = Prior.from_space(
        prior,
        center_values=prior._values,
        include_fidelity=include_fidelity,
    )
    inc_dist = Prior.from_space(
        inc,
        center_values=inc._values,
        include_fidelity=include_fidelity,
    )

    prior_score = prior_dist.pdf(encoded_config, frm=encoder).item()
    inc_score = inc_dist.pdf(encoded_config, frm=encoder).item()
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
