from __future__ import annotations

from itertools import product
from typing import Any, Literal

import torch

from neps.space import (
    Domain,
    HPOCategorical,
    HPOConstant,
    HPOFloat,
    HPOInteger,
    SearchSpace,
)

from neps import PipelineSpace, Categorical, Float, Integer, Fidelity
from neps.space.neps_spaces import neps_space
from neps.space.neps_spaces.sampling import RandomSampler


def make_grid(
    space: SearchSpace | PipelineSpace,
    *,
    size_per_numerical_hp: int = 10,
    ignore_fidelity: bool | Literal["highest fidelity"] = False
) -> list[dict[str, Any]]:
    """Get a grid of configurations from the search space.

    For [`Float`][neps.space.HPOFloat] and [`Integer`][neps.space.HPOInteger]
    the parameter `size_per_numerical_hp=` is used to determine a grid.

    For [`Categorical`][neps.space.HPOCategorical]
    hyperparameters, we include all the choices in the grid.

    For [`Constant`][neps.space.HPOConstant] hyperparameters,
    we include the constant value in the grid.

    Args:
        size_per_numerical_hp: The size of the grid for each numerical hyperparameter.

    Returns:
        A list of configurations from the search space.
    """
    param_ranges: dict[str, list[Any]] = {}
    if isinstance(space, SearchSpace):
        for name, hp in space.items():
            match hp:
                case HPOCategorical():
                    param_ranges[name] = list(hp.choices)
                case HPOConstant():
                    param_ranges[name] = [hp.value]
                case HPOInteger() | HPOFloat():
                    if hp.is_fidelity:
                        match ignore_fidelity:
                            case "highest fidelity":
                                param_ranges[name] = [hp.upper]
                                continue
                            case True:
                                param_ranges[name] = [hp.lower, hp.upper]
                            case False:
                                raise ValueError("Grid search does not support fidelity "
                                                 "natively. Please use the"
                                                 "ignore_fidelity parameter.")
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
        keys = list(space.keys())
        values = product(*param_ranges.values())
        return [dict(zip(keys, p, strict=False)) for p in values]
    elif isinstance(space, PipelineSpace):
        fid_ranges = {}
        for name, hp in space.get_attrs().items():
            match hp:
                case Categorical():
                    param_ranges[name] = range(len(hp.choices))
                case Fidelity():
                    if ignore_fidelity == "highest fidelity":
                        fid_ranges[name] = [hp.max_value]
                        continue
                    elif ignore_fidelity is True:
                        fid_ranges[name] = [hp.min_value, hp.max_value]
                    else:
                        raise ValueError("Grid search does not support fidelity natively." \
                        " Please use the ignore_fidelity parameter.")
                case Integer() | Float():
                    steps = size_per_numerical_hp
                    xs = torch.linspace(0, 1, steps=steps)
                    numeric_values = xs * (hp.max_value - hp.min_value) + hp.min_value
                    if isinstance(hp, Integer):
                        numeric_values = torch.round(numeric_values)
                    uniq_values = torch.unique(numeric_values).tolist()
                    param_ranges[name] = uniq_values
                case _:
                    raise NotImplementedError(f"Parameter type: {type(hp)}\n{hp} not supported yet in GridSearch")
        keys = list(param_ranges.keys())
        values = product(*param_ranges.values())
        config_dicts = [dict(zip(keys, p, strict=False)) for p in values]
        keys_fid = list(fid_ranges.keys())
        values_fid = product(*fid_ranges.values())
        fid_dicts = [dict(zip(keys_fid, p, strict=False)) for p in values_fid]
        configs = []
        random_config = neps_space.NepsCompatConverter.to_neps_config(neps_space.resolve(
                    pipeline=space,
                    domain_sampler=RandomSampler(predefined_samplings={}),
                    environment_values=fid_dicts[0],
                )[1])

        for config_dict in config_dicts:
            for fid_dict in fid_dicts:
                new_config = {}
                for param in random_config.keys():
                    for key in config_dict.keys():
                        if key in param:
                            new_config[param] = config_dict[key]
                    for key in fid_dict.keys():
                        if key in param:
                            new_config[param] = fid_dict[key]
                configs.append(new_config)
        return configs
