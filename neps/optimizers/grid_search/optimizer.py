from __future__ import annotations

import random
from collections.abc import Mapping
from itertools import product
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import torch

from neps.optimizers.base_optimizer import BaseOptimizer, SampledConfig
from neps.search_spaces.architecture.graph_grammar import GraphParameter
from neps.search_spaces.domain import UNIT_FLOAT_DOMAIN
from neps.search_spaces.hyperparameters.categorical import CategoricalParameter
from neps.search_spaces.hyperparameters.constant import ConstantParameter
from neps.search_spaces.hyperparameters.float import FloatParameter
from neps.search_spaces.hyperparameters.integer import IntegerParameter

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace
    from neps.state.optimizer import BudgetInfo
    from neps.state.trial import Trial


def _make_grid(
    space: SearchSpace,
    *,
    size_per_numerical_hp: int = 10,
) -> list[dict[str, Any]]:
    """Get a grid of configurations from the search space.

    For [`NumericalParameter`][neps.search_spaces.NumericalParameter] hyperparameters,
    the parameter `size_per_numerical_hp=` is used to determine a grid. If there are
    any duplicates, e.g. for an
    [`IntegerParameter`][neps.search_spaces.IntegerParameter], then we will
    remove duplicates.

    For [`CategoricalParameter`][neps.search_spaces.CategoricalParameter]
    hyperparameters, we include all the choices in the grid.

    For [`ConstantParameter`][neps.search_spaces.ConstantParameter] hyperparameters,
    we include the constant value in the grid.

    !!! note "TODO"

        Does not support graph parameters currently.

    !!! note "TODO"

        Include default hyperparameters in the grid.
        If all HPs have a `default` then add a single configuration.
        If only partial HPs have defaults then add all combinations of defaults, but
            only to the end of the list of configs.

    Args:
        size_per_numerical_hp: The size of the grid for each numerical hyperparameter.

    Returns:
        A list of configurations from the search space.
    """
    param_ranges: dict[str, list[Any]] = {}
    for name, hp in space.hyperparameters.items():
        match hp:
            # NOTE(eddiebergman): This is a temporary fix to avoid graphs
            # If this is resolved, please update the docstring!
            case GraphParameter():
                raise ValueError("Trying to create a grid for graphs!")
            case CategoricalParameter():
                param_ranges[name] = list(hp.choices)
            case ConstantParameter():
                param_ranges[name] = [hp.value]
            case IntegerParameter() | FloatParameter():
                if hp.is_fidelity:
                    param_ranges[name] = [hp.upper]
                    continue

                if hp.domain.cardinality is None:
                    steps = size_per_numerical_hp
                else:
                    steps = min(size_per_numerical_hp, hp.domain.cardinality)

                xs = torch.linspace(0, 1, steps=steps)
                numeric_values = hp.domain.cast(xs, frm=UNIT_FLOAT_DOMAIN)
                uniq_values = torch.unique(numeric_values).tolist()
                param_ranges[name] = uniq_values
            case _:
                raise NotImplementedError(f"Unknown Parameter type: {type(hp)}\n{hp}")

    values = product(*param_ranges.values())
    keys = list(space.hyperparameters.keys())

    return [dict(zip(keys, p, strict=False)) for p in values]


class GridSearch(BaseOptimizer):
    def __init__(self, pipeline_space: SearchSpace, seed: int | None = None):
        super().__init__(pipeline_space=pipeline_space)
        self.configs_list = _make_grid(pipeline_space)
        self.seed = seed

    @override
    def ask(
        self, trials: Mapping[str, Trial], budget_info: BudgetInfo | None
    ) -> SampledConfig:
        _num_previous_configs = len(trials)
        if _num_previous_configs > len(self.configs_list) - 1:
            raise ValueError("Grid search exhausted!")

        rng = random.Random(self.seed)
        configs = rng.sample(self.configs_list, len(self.configs_list))

        config = configs[_num_previous_configs]
        config_id = str(_num_previous_configs)
        return SampledConfig(config=config, id=config_id, previous_config_id=None)
