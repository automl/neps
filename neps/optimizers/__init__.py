from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Concatenate, Literal

from neps.optimizers.algorithms import (
    CustomOptimizer,
    OptimizerChoice,
    PredefinedOptimizers,
    determine_optimizer_automatically,
)
from neps.optimizers.optimizer import AskFunction, OptimizerInfo
from neps.utils.common import extract_keyword_defaults

if TYPE_CHECKING:
    from neps.space import SearchSpace
    from neps.space.neps_spaces.parameters import PipelineSpace


def _load_optimizer_from_string(
    optimizer: OptimizerChoice | Literal["auto"],
    space: SearchSpace | PipelineSpace,
    *,
    optimizer_kwargs: Mapping[str, Any] | None = None,
) -> tuple[AskFunction, OptimizerInfo]:
    if optimizer == "auto":
        _optimizer = determine_optimizer_automatically(space)
    else:
        _optimizer = optimizer

    optimizer_build = PredefinedOptimizers.get(_optimizer)
    if optimizer_build is None:
        raise ValueError(
            f"Unrecognized `optimizer` of type {type(optimizer)}."
            f" {optimizer}. Available optimizers are:"
            f" {PredefinedOptimizers.keys()}"
        )

    keywords = extract_keyword_defaults(optimizer_build)
    optimizer_kwargs = optimizer_kwargs or {}
    opt = optimizer_build(space, **optimizer_kwargs)  # type: ignore
    info = OptimizerInfo(name=_optimizer, info={**keywords, **optimizer_kwargs})
    return opt, info


def load_optimizer(
    optimizer: (
        OptimizerChoice
        | Mapping[str, Any]
        | tuple[OptimizerChoice, Mapping[str, Any]]
        | Callable[Concatenate[SearchSpace, ...], AskFunction]  # Hack, while we transit
        | Callable[Concatenate[PipelineSpace, ...], AskFunction]  # from SearchSpace to
        | Callable[Concatenate[SearchSpace | PipelineSpace, ...], AskFunction]  # Pipeline
        | CustomOptimizer
        | Literal["auto"]
    ),
    space: SearchSpace | PipelineSpace,
) -> tuple[AskFunction, OptimizerInfo]:
    match optimizer:
        # Predefined string (including "auto")
        case str():
            return _load_optimizer_from_string(optimizer, space)

        # Predefined string with kwargs
        case (opt, kwargs) if isinstance(opt, str):
            return _load_optimizer_from_string(opt, space, optimizer_kwargs=kwargs)  # type: ignore

        # Mapping with a name
        case {"name": name, **_kwargs}:
            return _load_optimizer_from_string(name, space, optimizer_kwargs=_kwargs)  # type: ignore

        # Provided optimizer initializer
        case _ if callable(optimizer):
            inner_optimizer = None
            if isinstance(optimizer, partial):
                inner_optimizer = optimizer.func
                while isinstance(inner_optimizer, partial):
                    inner_optimizer = inner_optimizer.func
            else:
                inner_optimizer = optimizer
            keywords = extract_keyword_defaults(optimizer)

            # Error catch and type ignore needed while we transition from SearchSpace to
            # Pipeline
            try:
                _optimizer = inner_optimizer(space, **keywords)  # type: ignore
            except TypeError as e:
                raise TypeError(
                    f"Optimizer {inner_optimizer} does not accept a space of type"
                    f" {type(space)}."
                ) from e

            info = OptimizerInfo(name=inner_optimizer.__name__, info=keywords)

            return _optimizer, info

        # Custom optimizer, we create it
        case CustomOptimizer(initialized=False):
            _optimizer = optimizer.create(space)
            keywords = extract_keyword_defaults(optimizer.optimizer)
            info = OptimizerInfo(
                name=optimizer.name, info={**keywords, **optimizer.kwargs}
            )
            return _optimizer, info

        # Custom (already initialized) optimizer
        case CustomOptimizer(initialized=True):
            preinit_opt = optimizer.optimizer
            info = OptimizerInfo(name=optimizer.name, info=optimizer.kwargs)
            return preinit_opt, info  # type: ignore

        case _:
            raise ValueError(
                f"Unrecognized `optimizer` of type {type(optimizer)}."
                f" {optimizer}. Must either be a string, callable or"
                " a `CustomOptimizer` instance."
            )
