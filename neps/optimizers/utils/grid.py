from __future__ import annotations

from itertools import product
from typing import Any, Literal

import torch

from neps import Categorical, Fidelity, Float, Integer, PipelineSpace
from neps.space import (
    Domain,
    HPOCategorical,
    HPOConstant,
    HPOFloat,
    HPOInteger,
    SearchSpace,
)
from neps.space.neps_spaces import neps_space
from neps.space.neps_spaces.sampling import RandomSampler


def make_grid(  # noqa: PLR0912, PLR0915, C901
    space: SearchSpace | PipelineSpace,
    *,
    size_per_numerical_hp: int | dict[str, int] = 5,
    ignore_fidelity: bool | Literal["highest_fidelity"] = False,
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
            Can be an integer (applies to all numerical hyperparameters) or a dictionary
            mapping parameter names to their grid sizes.

    Returns:
        A list of configurations from the search space.
    """
    param_ranges: dict[str, list[Any]] = {}
    
    # Normalize size_per_numerical_hp to a dictionary
    if isinstance(size_per_numerical_hp, int):
        size_mapping: dict[str, int] = {}
        default_size = size_per_numerical_hp
    else:
        size_mapping = size_per_numerical_hp
        default_size = 10
        
        # Validate that all keys in size_mapping correspond to numerical parameters
        if isinstance(space, SearchSpace):
            numerical_params = {
                name for name, hp in space.items()
                if isinstance(hp, (HPOInteger, HPOFloat))
            }
            invalid_keys = set(size_mapping.keys()) - numerical_params
            if invalid_keys:
                raise ValueError(
                    f"size_per_numerical_hp dictionary contains invalid keys: {invalid_keys}. "
                    f"Only numerical parameters (HPOInteger, HPOFloat) are allowed. "
                    f"Valid numerical parameters are: {numerical_params}"
                )
        elif isinstance(space, PipelineSpace):
            numerical_params = {
                name for name, hp in space.get_attrs().items()
                if isinstance(hp, (Integer, Float))
            }
            invalid_keys = set(size_mapping.keys()) - numerical_params
            if invalid_keys:
                raise ValueError(
                    f"size_per_numerical_hp dictionary contains invalid keys: {invalid_keys}. "
                    f"Only numerical parameters (Integer, Float) are allowed. "
                    f"Valid numerical parameters are: {numerical_params}"
                )
        
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
                            case "highest_fidelity":
                                param_ranges[name] = [hp.upper]
                                continue
                            case True:
                                param_ranges[name] = [hp.lower, hp.upper]
                            case False:
                                raise ValueError(
                                    "Grid search does not support fidelity "
                                    "natively. Please use the"
                                    "ignore_fidelity parameter."
                                )
                    
                    # Get grid size for this parameter
                    steps = size_mapping.get(name, default_size)
                    
                    if hp.domain.cardinality is None:
                        steps = steps
                    else:
                        steps = min(steps, hp.domain.cardinality)

                    xs = torch.linspace(0, 1, steps=steps)
                    numeric_values = hp.domain.cast(xs, frm=Domain.unit_float())
                    uniq_values = torch.unique(numeric_values).tolist()
                    param_ranges[name] = uniq_values
                case _:
                    raise NotImplementedError(f"Unknown Parameter type: {type(hp)}\n{hp}")
        keys = list(space.keys())
        values = product(*param_ranges.values())
        configs = [dict(zip(keys, p, strict=False)) for p in values]
        return configs
    if isinstance(space, PipelineSpace):
        fid_ranges: dict[str, list[float]] = {}
        for name, hp in space.get_attrs().items():
            if isinstance(hp, Categorical):
                if isinstance(hp.choices, tuple):  # type: ignore[unreachable]
                    param_ranges[name] = list(range(len(hp.choices)))
                else:
                    raise NotImplementedError(
                        "Grid search only supports categorical choices as tuples."
                    )
            elif isinstance(hp, Fidelity):
                if ignore_fidelity == "highest_fidelity":  # type: ignore[unreachable]
                    fid_ranges[name] = [hp.upper]
                    continue
                if ignore_fidelity is True:
                    fid_ranges[name] = [hp.lower, hp.upper]
                    continue
                raise ValueError(
                    "Grid search does not support fidelity natively."
                    " Please use the ignore_fidelity parameter."
                )
            elif isinstance(hp, Integer | Float):
                # Get grid size for this parameter
                steps = size_mapping.get(name, default_size)  # type: ignore[unreachable]
                xs = torch.linspace(0, 1, steps=steps)
                numeric_values = xs * (hp.upper - hp.lower) + hp.lower
                if isinstance(hp, Integer):
                    numeric_values = torch.round(numeric_values)
                uniq_values = torch.unique(numeric_values).tolist()
                param_ranges[name] = uniq_values
            else:
                raise NotImplementedError(
                    f"Parameter type: {type(hp)}\n{hp} not supported yet in GridSearch"
                )
        keys = list(param_ranges.keys())
        values = product(*param_ranges.values())
        config_dicts = [dict(zip(keys, p, strict=False)) for p in values]
        keys_fid = list(fid_ranges.keys())
        values_fid = product(*fid_ranges.values())
        fid_dicts = [dict(zip(keys_fid, p, strict=False)) for p in values_fid]
        configs = []
        random_config = neps_space.NepsCompatConverter.to_neps_config(
            neps_space.resolve(
                pipeline=space,
                domain_sampler=RandomSampler(predefined_samplings={}),
                environment_values=fid_dicts[0],
            )[1]
        )

        for config_dict in config_dicts:
            for fid_dict in fid_dicts:
                new_config = {}
                for param in random_config:
                    for key in config_dict:
                        if key in param:
                            new_config[param] = config_dict[key]
                    for key in fid_dict:
                        if key in param:
                            new_config[param] = fid_dict[key]
                configs.append(new_config)
        return configs

    raise TypeError(
        f"Unsupported space type: {type(space)}"
    )  # More informative than None
