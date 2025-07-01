from __future__ import annotations

from itertools import product
from typing import Any

import torch

from neps.space import Categorical, Constant, Domain, Float, Integer, SearchSpace


def make_grid(
    space: SearchSpace,
    *,
    size_per_numerical_hp: int = 10,
    ignore_fidelity: bool = True,
) -> list[dict[str, Any]]:
    """Get a grid of configurations from the search space.

    For [`Float`][neps.space.Float] and [`Integer`][neps.space.Integer]
    the parameter `size_per_numerical_hp=` is used to determine a grid.

    For [`Categorical`][neps.space.Categorical]
    hyperparameters, we include all the choices in the grid.

    For [`Constant`][neps.space.Constant] hyperparameters,
    we include the constant value in the grid.

    Args:
        size_per_numerical_hp: The size of the grid for each numerical hyperparameter.

    Returns:
        A list of configurations from the search space.
    """
    param_ranges: dict[str, list[Any]] = {}
    for name, hp in space.items():
        match hp:
            case Categorical():
                param_ranges[name] = list(hp.choices)
            case Constant():
                param_ranges[name] = [hp.value]
            case Integer() | Float():
                if hp.is_fidelity and ignore_fidelity:
                    param_ranges[name] = [hp.upper]
                    continue

                if hp.domain.cardinality is None:
                    steps = size_per_numerical_hp
                else:
                    steps = min(size_per_numerical_hp, hp.domain.cardinality)

                xs = torch.linspace(0, 1, steps=steps)
                numeric_values = hp.domain.cast(xs, frm=Domain.unit_float())
                uniq_values = torch.unique(numeric_values).tolist()
                param_ranges[name] = uniq_values
            case _:
                raise NotImplementedError(f"Unknown Parameter type: {type(hp)}\n{hp}")
    values = product(*param_ranges.values())
    keys = list(space.keys())

    return [dict(zip(keys, p, strict=False)) for p in values]
