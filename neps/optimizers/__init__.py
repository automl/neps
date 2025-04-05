from __future__ import annotations

from collections.abc import Callable, Mapping
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


def _load_optimizer_from_string(
    optimizer: OptimizerChoice | Literal["auto"],
    space: SearchSpace,
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
    if space.fidelity is not None and not optimizer_build.Supports.fidelity:
        raise ValueError(
            f"Optimizer `{_optimizer}` does not support fidelity. "
            f"Please choose a different optimizer or consider "
            f"setting the fidelity to a constant. Got: {space.fidelity}."
        )
    if space.fidelity is None and optimizer_build.Supports.fidelity:
        raise ValueError(
            f"Optimizer `{_optimizer}` requires fidelity. "
            "Please choose a different optimizer."
        )
    if (
        any(element.prior is not None for element in list(space.elements.values()))
        and not optimizer_build.Supports.uses_priors
    ):
        raise ValueError(
            f"Optimizer `{_optimizer}` does not support priors. "
            f"Please choose a different optimizer. Got: "
            f"{
                [
                    element
                    for element in space.elements.values()
                    if element.prior is not None
                ]
            }."
        )
    if (
        not any(element.prior is not None for element in list(space.elements.values()))
        and optimizer_build.Supports.requires_priors
    ):
        raise ValueError(
            f"Optimizer `{_optimizer}` requires prior. "
            "Please choose a different optimizer."
        )

    keywords = extract_keyword_defaults(optimizer_build)
    optimizer_kwargs = optimizer_kwargs or {}
    opt = optimizer_build(space, **optimizer_kwargs)
    info = OptimizerInfo(name=_optimizer, info={**keywords, **optimizer_kwargs})
    return opt, info


def load_optimizer(
    optimizer: (
        OptimizerChoice
        | Mapping[str, Any]
        | tuple[OptimizerChoice, Mapping[str, Any]]
        | Callable[Concatenate[SearchSpace, ...], AskFunction]
        | CustomOptimizer
        | Literal["auto"]
    ),
    space: SearchSpace,
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
            keywords = extract_keyword_defaults(optimizer)
            _optimizer = optimizer(space)
            info = OptimizerInfo(name=optimizer.__name__, info=keywords)
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
