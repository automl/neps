from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Concatenate, Literal, TypeAlias

from autoscan.optimizers.algorithms import (
    CustomOptimizer,
    OptimizerChoice,
    PredefinedOptimizers,
    determine_optimizer_automatically,
)
from autoscan.optimizers.optimizer import AskFunction, OptimizerInfo, ScalingOptimizer
from autoscan.utils.common import extract_keyword_defaults
from autoscan.utils.files import serializable_format

if TYPE_CHECKING:
    from autoscan.space import SearchSpace
    from autoscan.space.autoscan_spaces.parameters import PipelineSpace
    from autoscan.utils.cost_estimator import CostEstimator

    # Every form an optimizer can be specified in
    OptimizerSpec: TypeAlias = (
        OptimizerChoice
        | Mapping[str, Any]
        | tuple[OptimizerChoice, Mapping[str, Any]]
        | Callable[Concatenate[SearchSpace, ...], AskFunction]  # Hack, while we transit
        | Callable[Concatenate[PipelineSpace, ...], AskFunction]  # from SearchSpace to
        | Callable[Concatenate[SearchSpace | PipelineSpace, ...], AskFunction]  # Pipeline
        | CustomOptimizer
        | Literal["auto"]
    )

logger = logging.getLogger(__name__)


def load_optimizer(
    optimizer: OptimizerSpec,
    space: SearchSpace | PipelineSpace,
    *,
    cost_estimator: CostEstimator | None = None,
    max_cost_per_config: float | None = None,
) -> tuple[AskFunction, OptimizerInfo]:
    """Create the optimizer and the info describing it.

    Args:
        optimizer: Which optimizer to create.
        space: The space the optimizer searches.
        cost_estimator: Handed to optimizers inheriting from `ScalingOptimizer`.
        max_cost_per_config: The cost ceiling that goes with `cost_estimator`.
    """
    _optimizer, info = _construct_optimizer(optimizer, space)
    if cost_estimator is not None:
        _attach_cost_estimator(
            _optimizer,
            info,
            cost_estimator=cost_estimator,
            max_cost_per_config=max_cost_per_config,
        )
    return _optimizer, info


def _yaml_safe(value: Any) -> Any:
    value = serializable_format(value)
    match value:
        case None | bool() | int() | float() | str():
            return value
        case list():
            return [_yaml_safe(v) for v in value]
        case dict():
            return {k: _yaml_safe(v) for k, v in value.items()}
        case _:
            obj = value if hasattr(value, "__qualname__") else type(value)
            return f"{obj.__module__}.{obj.__qualname__}"


def _make_info(
    name: str, info: Mapping[str, Any], optimizer: Any = None
) -> OptimizerInfo:
    optimizer_info = OptimizerInfo(name=name, info=_yaml_safe(dict(info)))
    if derived := getattr(optimizer, "derived_info", None):
        optimizer_info["derived"] = _yaml_safe(dict(derived))
    return optimizer_info


def _resolve_kwargs(optimizer: Callable, space: Any) -> dict[str, Any]:
    func = optimizer
    args: tuple[Any, ...] = ()
    keywords: dict[str, Any] = {}
    while isinstance(func, partial):
        args = (*func.args, *args)
        keywords = {**func.keywords, **keywords}
        func = func.func

    try:
        bound = inspect.signature(func).bind_partial(*args, space, **keywords)
    except (TypeError, ValueError):
        return extract_keyword_defaults(optimizer)
    bound.apply_defaults()

    resolved: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if value is space:
            continue
        match bound.signature.parameters[name].kind:
            case inspect.Parameter.VAR_KEYWORD:
                resolved.update(value)
            case inspect.Parameter.VAR_POSITIONAL:
                if extra := [v for v in value if v is not space]:
                    resolved[name] = extra
            case _:
                resolved[name] = value
    return resolved


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
    optimizer_kwargs = dict(optimizer_kwargs)  # Make mutable copy
    if _optimizer == "primo":
        optimizer_kwargs["prior_centers"] = optimizer_kwargs.get("prior_centers", {})
    opt = optimizer_build(space, **optimizer_kwargs)  # type: ignore
    info = _make_info(_optimizer, {**keywords, **optimizer_kwargs}, opt)
    return opt, info


def _construct_optimizer(
    optimizer: OptimizerSpec, space: SearchSpace | PipelineSpace
) -> tuple[AskFunction, OptimizerInfo]:
    """Build the optimizer from whatever form the user specified it in."""
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
            inner_optimizer = optimizer
            while isinstance(inner_optimizer, partial):
                inner_optimizer = inner_optimizer.func
            # Callable instances have no `__name__`, so fall back to their class name
            name = getattr(inner_optimizer, "__name__", type(inner_optimizer).__name__)
            if name == "<lambda>":
                logger.warning(
                    "The optimizer was given as a lambda, so its settings cannot be"
                    " recorded in optimizer_info.yaml. Pass a function, a"
                    " `functools.partial` or `autoscan.algorithms.custom(...)` instead."
                )
            keywords = _resolve_kwargs(optimizer, space)

            # Error catch and type ignore needed while we transition from SearchSpace to
            # Pipeline
            try:
                _optimizer = optimizer(space)  # type: ignore
            except TypeError as e:
                raise TypeError(
                    f"Optimizer {inner_optimizer} does not accept a space of type"
                    f" {type(space)}."
                ) from e

            return _optimizer, _make_info(name, keywords, _optimizer)

        # Custom optimizer, we create it
        case CustomOptimizer(initialized=False):
            _optimizer = optimizer.create(space)
            keywords = _resolve_kwargs(
                partial(optimizer.optimizer, **optimizer.kwargs), space
            )
            return _optimizer, _make_info(optimizer.name, keywords, _optimizer)

        # Custom (already initialized) optimizer
        case CustomOptimizer(initialized=True):
            preinit_opt = optimizer.optimizer
            info = _make_info(optimizer.name, optimizer.kwargs, preinit_opt)
            return preinit_opt, info  # type: ignore

        case _:
            raise ValueError(
                f"Unrecognized `optimizer` of type {type(optimizer)}."
                f" {optimizer}. Must either be a string, callable or"
                " a `CustomOptimizer` instance."
            )


def _attach_cost_estimator(
    optimizer: AskFunction,
    info: OptimizerInfo,
    *,
    cost_estimator: CostEstimator,
    max_cost_per_config: float | None,
) -> None:
    """Hand the cost estimator of the run to an optimizer that can use it."""
    if isinstance(optimizer, ScalingOptimizer):
        optimizer.set_cost_estimator(
            cost_estimator, max_cost_per_config=max_cost_per_config
        )
    else:
        logger.warning(
            "The optimizer '%s' does not use `cost_estimator`%s. Optimizers that do"
            " inherit from `autoscan.optimizers.optimizer.ScalingOptimizer`.",
            info["name"],
            " and `max_cost_per_config`" if max_cost_per_config is not None else "",
        )

    # Not in `info`, which is passed back to the optimizer when continuing a run
    info["derived"] = {
        **info.get("derived", {}),
        "max_cost_per_config": max_cost_per_config,
    }
