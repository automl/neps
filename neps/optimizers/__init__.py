from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from neps.optimizers.algorithms import (
    OptimizerChoice,
    PredefinedOptimizers,
    determine_optimizer_automatically,
)
from neps.optimizers.optimizer import AskFunction  # noqa: TC001
from neps.utils.common import extract_keyword_defaults

if TYPE_CHECKING:
    from neps.space import SearchSpace


def _load_optimizer_from_string(
    optimizer: OptimizerChoice | Literal["auto"],
    space: SearchSpace,
    *,
    optimizer_kwargs: Mapping[str, Any] | None = None,
) -> tuple[AskFunction, dict[str, Any]]:
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

    info = extract_keyword_defaults(optimizer_build)
    info["name"] = _optimizer

    optimizer_kwargs = optimizer_kwargs or {}
    opt = optimizer_build(space, **optimizer_kwargs)
    return opt, info


def load_optimizer(
    optimizer: (
        OptimizerChoice
        | Mapping[str, Any]
        | tuple[OptimizerChoice | Callable[..., AskFunction], Mapping[str, Any]]
        | Callable[..., AskFunction]
        | Literal["auto"]
    ),
    space: SearchSpace,
) -> tuple[AskFunction, dict[str, Any]]:
    match optimizer:
        # Predefined string
        case str():
            return _load_optimizer_from_string(optimizer, space)

        # class/builder
        case _ if callable(optimizer):
            info = extract_keyword_defaults(optimizer)
            _optimizer = optimizer(space)
            info["name"] = optimizer.__name__
            return _optimizer, info

        # Predefined string with kwargs
        case (opt, kwargs) if isinstance(opt, str):
            return _load_optimizer_from_string(opt, space, optimizer_kwargs=kwargs)  # type: ignore

        # class/builder with kwargs
        case (opt, kwargs):
            info = extract_keyword_defaults(opt)  # type: ignore
            info["name"] = opt.__name__  # type: ignore
            _optimizer = opt(space, **kwargs)  # type: ignore
            return _optimizer, info

        # Mapping with a name
        case {"name": name, **_kwargs}:
            return _load_optimizer_from_string(name, space, optimizer_kwargs=_kwargs)  # type: ignore

        case _:
            raise ValueError(
                f"Unrecognized `optimizer` of type {type(optimizer)}."
                f" {optimizer}. Must either be a string or a callable."
            )
