from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, Literal, TypeAlias

import torch

from neps.optimizers.bo import BayesianOptimization
from neps.optimizers.bracket_optimizer import BracketOptimizer
from neps.optimizers.grid_search import GridSearch
from neps.optimizers.ifbo import IFBO
from neps.optimizers.models.ftpfn import FTPFNSurrogate
from neps.optimizers.optimizer import AskFunction  # noqa: TC001
from neps.optimizers.priorband import PriorBandArgs
from neps.optimizers.random_search import RandomSearch
from neps.sampling import Prior, Sampler, Uniform
from neps.search_spaces.encoding import CategoricalToUnitNorm, ConfigEncoder
from neps.utils.common import extract_keyword_defaults

if TYPE_CHECKING:
    from neps.search_spaces.search_space import SearchSpace


def _bo(
    pipeline_space: SearchSpace,
    *,
    initial_design_size: int | Literal["ndim"] = "ndim",
    use_priors: bool,
    cost_aware: bool | Literal["log"],
    sample_prior_first: bool,
    device: torch.device | str | None,
) -> BayesianOptimization:
    """Initialise the BO loop.

    Args:
        pipeline_space: Space in which to search
        initial_design_size: Number of samples used before using the surrogate model.
            If "ndim", it will use the number of parameters in the search space.
        use_priors: Whether to use priors set on the hyperparameters during search.
        cost_aware: Whether to consider reported "cost" from configurations in decision
            making. If True, the optimizer will weigh potential candidates by how much
            they cost, incentivising the optimizer to explore cheap, good performing
            configurations. This amount is modified over time. If "log", the cost
            will be log-transformed before being used.

            !!! warning

                If using `cost`, cost must be provided in the reports of the trials.

        sample_prior_first: Whether to sample the default configuration first.
        device: Device to use for the optimization.

    Raises:
        ValueError: if initial_design_size < 1
    """
    if any(pipeline_space.graphs):
        raise NotImplementedError("Only supports flat search spaces for now!")

    if any(pipeline_space.fidelities):
        raise ValueError(
            "Fidelities are not supported for BayesianOptimization."
            " Please consider setting the fidelity to a constant value."
            f" Got: {pipeline_space.fidelities}"
        )

    match initial_design_size:
        case "ndim":
            n_initial_design_size = len(pipeline_space.numerical) + len(
                pipeline_space.categoricals
            )
        case int():
            if initial_design_size < 1:
                raise ValueError("initial_design_size should be greater than 0")

            n_initial_design_size = initial_design_size
        case _:
            raise ValueError(
                "initial_design_size should be either 'ndim' or a positive integer"
            )

    match device:
        case str():
            device = torch.device(device)
        case None:
            device = torch.get_default_device()
        case torch.device():
            pass
        case _:
            raise ValueError("device should be a string, torch.device or None")

    return BayesianOptimization(
        pipeline_space=pipeline_space,
        encoder=ConfigEncoder.from_space(space=pipeline_space),
        n_initial_design=n_initial_design_size,
        cost_aware=cost_aware,
        prior=Prior.from_space(pipeline_space) if use_priors is True else None,
        sample_prior_first=sample_prior_first,
        device=device,
    )


def _bracket_optimizer(  # noqa: C901
    pipeline_space: SearchSpace,
    *,
    bracket_type: Literal["successive_halving", "hyperband", "asha", "async_hb"],
    eta: int,
    sampler: Literal["uniform", "prior", "priorband"],
    sample_prior_first: bool | Literal["highest_fidelity"],
    # NOTE: This is the only argument to get a default, since it
    # is not required for hyperband style algorithms, only single bracket
    # style ones.
    early_stopping_rate: int | None = None,
) -> BracketOptimizer:
    """Initialise a bracket optimizer.

    Args:
        pipeline_space: Space in which to search
        bracket_type: The type of bracket to use. Can be one of:

            * "successive_halving": Successive Halving
            * "hyperband": HyperBand
            * "asha": ASHA
            * "async_hb": Async

        eta: The reduction factor used for building brackets
        early_stopping_rate: Determines the number of rungs in a bracket
            Choosing 0 creates maximal rungs given the fidelity bounds.

            !!! warning

                This is only used for Successive Halving and Asha. If set
                to not `None`, then the bracket type must be one of those.

        sampler: The type of sampling procedure to use:

            * If "uniform", samples uniformly from the space when it needs to sample
            * If "prior", samples from the prior distribution built from the prior
              and prior_confidence values in the pipeline space.
            * If "priorband", samples with weights according to the PriorBand
                algorithm. See: https://arxiv.org/abs/2306.12370

        sample_prior_first: Whether to sample the prior configuration first.
    """
    assert pipeline_space.fidelity is not None
    assert pipeline_space.fidelity_name is not None
    if len(pipeline_space.fidelities) != 1:
        raise ValueError(
            "Fidelity should be defined in the pipeline space."
            f"\nGot: {pipeline_space.fidelities}"
        )

    if sample_prior_first not in (True, False, "highest_fidelity"):
        raise ValueError(
            "sample_prior_first should be either True, False or 'highest_fidelity'"
        )

    from neps.optimizers.utils import brackets

    match bracket_type:
        case "successive_halving":
            assert early_stopping_rate is not None
            rung_to_fidelity, rung_sizes = brackets.calculate_sh_rungs(
                bounds=(pipeline_space.fidelity.lower, pipeline_space.fidelity.upper),
                eta=eta,
                early_stopping_rate=early_stopping_rate,
            )
            create_brackets = partial(
                brackets.Sync.create_repeating, rung_sizes=rung_sizes
            )
        case "hyperband":
            assert early_stopping_rate is None
            rung_to_fidelity, bracket_layouts = brackets.calculate_hb_bracket_layouts(
                bounds=(pipeline_space.fidelity.lower, pipeline_space.fidelity.upper),
                eta=eta,
            )
            create_brackets = partial(
                brackets.Hyperband.create_repeating,
                bracket_layouts=bracket_layouts,
            )
        case "asha":
            assert early_stopping_rate is not None
            rung_to_fidelity, _rung_sizes = brackets.calculate_sh_rungs(
                bounds=(pipeline_space.fidelity.lower, pipeline_space.fidelity.upper),
                eta=eta,
                early_stopping_rate=early_stopping_rate,
            )
            create_brackets = partial(
                brackets.Async.create, rungs=list(rung_to_fidelity), eta=eta
            )
        case "async_hb":
            assert early_stopping_rate is None
            rung_to_fidelity, bracket_layouts = brackets.calculate_hb_bracket_layouts(
                bounds=(pipeline_space.fidelity.lower, pipeline_space.fidelity.upper),
                eta=eta,
            )
            # We don't care about the capacity of each bracket, we need the rung layout
            bracket_rungs = [list(bracket.keys()) for bracket in bracket_layouts]
            create_brackets = partial(
                brackets.AsyncHyperband.create,
                bracket_rungs=bracket_rungs,
                eta=eta,
            )
        case _:
            raise ValueError(f"Unknown bracket type: {bracket_type}")

    encoder = ConfigEncoder.from_space(pipeline_space, include_fidelity=False)

    match sampler:
        case "uniform":
            _sampler = Sampler.uniform(ndim=encoder.ndim)
        case "prior":
            _sampler = Prior.from_config(
                pipeline_space.prior_config, space=pipeline_space
            )
        case "priorband":
            _sampler = PriorBandArgs(mutation_rate=0.5, mutation_std=0.25)
        case PriorBandArgs() | Sampler():
            _sampler = sampler
        case _:
            raise ValueError(f"Unknown sampler: {sampler}")

    return BracketOptimizer(
        pipeline_space=pipeline_space,
        encoder=encoder,
        eta=eta,
        rung_to_fid=rung_to_fidelity,
        fid_min=pipeline_space.fidelity.lower,
        fid_max=pipeline_space.fidelity.upper,
        fid_name=pipeline_space.fidelity_name,
        sampler=_sampler,
        sample_prior_first=sample_prior_first,
        create_brackets=create_brackets,
    )


def _grid_search(pipeline_space: SearchSpace) -> GridSearch:
    from neps.optimizers.utils.grid import make_grid

    return GridSearch(
        pipeline_space=pipeline_space,
        configs_list=make_grid(pipeline_space),
    )


def _random_search(
    pipeline_space: SearchSpace,
    *,
    use_priors: bool = False,
    ignore_fidelity: bool = True,
) -> RandomSearch:
    """Initialize the random search optimizer.

    Args:
        pipeline_space: The search space to sample from.
        use_priors: Whether to use priors when sampling.
        ignore_fidelity: Whether to ignore fidelity when sampling.
            In this case, the max fidelity is always used.
        seed: The seed for the random number generator.
    """
    encoder = ConfigEncoder.from_space(
        pipeline_space, include_fidelity=not ignore_fidelity
    )

    if use_priors:
        sampler = Prior.from_space(pipeline_space, include_fidelity=not ignore_fidelity)
    else:
        sampler = Uniform(ndim=encoder.ndim)

    return RandomSearch(
        pipeline_space=pipeline_space,
        encoder=encoder,
        sampler=sampler,
        ignore_fidelity=ignore_fidelity,
    )


def _ifbo(
    pipeline_space: SearchSpace,
    *,
    step_size: int | float = 1,
    use_priors: bool = False,
    sample_prior_first: bool = False,
    initial_design_size: int | Literal["ndim"] = "ndim",
    device: torch.device | str | None = None,
    surrogate_path: str | Path | None = None,
    surrogate_version: str = "0.0.1",
) -> IFBO:
    """Initialise.

    Args:
        pipeline_space: Space in which to search
        step_size: The size of the step to take in the fidelity domain.
        sampling_policy: The type of sampling procedure to use
        promotion_policy: The type of promotion procedure to use
        sample_prior_first: Whether to sample the default configuration first
        initial_design_size: Number of configs to sample before starting optimization

            If None, the number of configs will be equal to the number of dimensions.

        device: Device to use for the model
    """
    from neps.optimizers.ifbo import adjust_pipeline_space_to_match_stepsize

    # TODO: I'm not sure how this might effect tables, whose lowest fidelity
    # might be below to possibly increased lower bound.
    space, fid_bins = adjust_pipeline_space_to_match_stepsize(pipeline_space, step_size)
    assert space.fidelity is not None
    assert isinstance(space.fidelity_name, str)

    match initial_design_size:
        case "ndim":
            _initial_design_size = len(space.numerical) + len(space.categoricals)
        case _:
            _initial_design_size = initial_design_size

    match device:
        case str():
            device = torch.device(device)
        case None:
            device = torch.get_default_device()
        case torch.device():
            pass
        case _:
            raise ValueError("device should be a string, torch.device or None")

    return IFBO(
        pipeline_space=pipeline_space,
        n_fidelity_bins=fid_bins,
        device=device,
        sample_prior_first=sample_prior_first,
        n_initial_design=_initial_design_size,
        fid_domain=space.fidelity.domain,
        fidelity_name=space.fidelity_name,
        prior=(Prior.from_space(space, include_fidelity=False) if use_priors else None),
        ftpfn=FTPFNSurrogate(
            target_path=Path(surrogate_path) if surrogate_path is not None else None,
            version=surrogate_version,
            device=device,
        ),
        encoder=ConfigEncoder.from_space(
            space=space,
            # FTPFN doesn't support categoricals and we were recomended
            # to just evenly distribute in the unit norm
            custom_transformers={
                cat_name: CategoricalToUnitNorm(choices=cat.choices)
                for cat_name, cat in space.categoricals.items()
            },
        ),
    )


# NOTE: We wrap all of these in partials so we have access to the default
# parameters that they would be created with. These parameters are all
# easy to serialize and so can be written to yaml for optimizer info.
random_search = partial(_random_search, use_priors=False, ignore_fidelity=True)
grid_search = partial(_grid_search)
ifbo = partial(
    _ifbo,
    step_size=1,
    use_priors=False,
    sample_prior_first=False,
    initial_design_size="ndim",
    device=None,
    surrogate_path=None,
    surrogate_version="0.0.1",
)
hyperband = partial(
    _bracket_optimizer,
    bracket_type="hyperband",
    eta=3,
    sampler="uniform",
    sample_prior_first=False,
)
hyperband_prior = partial(
    _bracket_optimizer,
    sampler="prior",
    bracket_type="hyperband",
    eta=3,
    sample_prior_first=False,
)
successive_halving = partial(
    _bracket_optimizer,
    bracket_type="successive_halving",
    eta=3,
    early_stopping_rate=0,
    sampler="uniform",
    sample_prior_first=False,
)
successive_halving_prior = partial(
    _bracket_optimizer,
    sampler="prior",
    bracket_type="successive_halving",
    eta=3,
    early_stopping_rate=0,
    sample_prior_first=False,
)
asha = partial(
    _bracket_optimizer,
    bracket_type="asha",
    eta=3,
    early_stopping_rate=0,
    sampler="uniform",
    sample_prior_first=False,
)
asha_prior = partial(
    _bracket_optimizer,
    sampler="prior",
    bracket_type="asha",
    eta=3,
    early_stopping_rate=0,
    sample_prior_first=False,
)
async_hb = partial(
    _bracket_optimizer,
    bracket_type="async_hb",
    eta=3,
    sampler="uniform",
    sample_prior_first=False,
)
async_hb_prior = partial(
    _bracket_optimizer,
    sampler="prior",
    bracket_type="async_hb",
    eta=3,
    sample_prior_first=False,
)
priorband = partial(
    _bracket_optimizer,
    sampler="priorband",
    bracket_type="hyperband",
    eta=3,
    sample_prior_first=False,
)
priorband_sh = partial(
    _bracket_optimizer,
    sampler="priorband",
    bracket_type="successive_halving",
    eta=3,
    early_stopping_rate=0,
    sample_prior_first=False,
)
priorband_asha = partial(
    _bracket_optimizer,
    sampler="priorband",
    bracket_type="asha",
    eta=3,
    early_stopping_rate=0,
    sample_prior_first=False,
)
priorband_async = partial(
    _bracket_optimizer,
    sampler="priorband",
    bracket_type="async_hb",
    eta=3,
    sample_prior_first=False,
)
pibo = partial(
    _bo,
    use_priors=True,
    initial_design_size="ndim",
    cost_aware=False,
    sample_prior_first=False,
    device=None,
)
bayesian_optimization = partial(
    _bo,
    use_priors=False,
    initial_design_size="ndim",
    cost_aware=False,
    sample_prior_first=False,
    device=None,
)
bayesian_optimization_cost_aware = partial(
    _bo,
    use_priors=False,
    initial_design_size="ndim",
    cost_aware=True,
    sample_prior_first=False,
    device=None,
)


def determine_optimizer_automatically(space: SearchSpace) -> str:
    if space.has_prior:
        return "priorband" if len(space.fidelities) > 0 else "pibo"

    return "hyperband" if len(space.fidelities) > 0 else "bayesian_optimization"


# NOTE: If updating this, you should update to more place...
# 1. The `OptimizerChoice` just below this.
# 2. The `run()` function documentation
PredefinedOptimizers: Mapping[
    str,
    Callable[Concatenate[SearchSpace, ...], AskFunction],
] = {
    # BO kind
    "bayesian_optimization": bayesian_optimization,
    "bayesian_optimization_cost_aware": bayesian_optimization_cost_aware,
    "bayesian_optimization_prior": pibo,
    "pibo": pibo,
    # Successive Halving
    "successive_halving": successive_halving,
    "successive_halving_prior": successive_halving_prior,
    # Hyperband
    "hyperband": hyperband,
    "hyperband_prior": hyperband_prior,
    # ASHA
    "asha": asha,
    "asha_prior": asha_prior,
    # AsyncHB
    "async_hb": async_hb,
    "async_hb_prior": async_hb_prior,
    # Priorband
    "priorband": priorband,
    "priorband_sh": priorband_sh,
    "priorband_asha": priorband_asha,
    "priorband_async": priorband_async,
    # Model based hyperband Other
    "random_search": random_search,
    "grid_search": grid_search,
    "ifbo": ifbo,
    # TODO:
    # "mobster": MOBSTER,
    # "priorband_bo": partial(PriorBand, model_based=True),
}

OptimizerChoice: TypeAlias = Literal[
    "bayesian_optimization"
    "bayesian_optimization_cost_aware"
    "pibo"
    # Successive Halving
    "successive_halving"
    "successive_halving_prior"
    # Hyperband
    "hyperband"
    "hyperband_prior"
    # ASHA
    "asha"
    "asha_prior"
    # AsyncHB
    "async_hb"
    "async_hb_prior"
    # Priorband
    "priorband"
    "priorband_sh"
    "priorband_asha"
    "priorband_async"
    # Other
    "random_search"
    "grid_search"
    "ifbo"
]


def _load_optimizer_from_string(
    optimizer: OptimizerChoice,
    space: SearchSpace,
    *,
    optimizer_kwargs: Mapping[str, Any] | None = None,
) -> tuple[AskFunction, dict[str, Any]]:
    if optimizer == "auto":
        _optimizer = determine_optimizer_automatically(space)
    else:
        _optimizer = optimizer

    optimizer_build = PredefinedOptimizers.get(_optimizer)
    print(optimizer_build)  # noqa: T201
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
    ),
    space: SearchSpace,
) -> tuple[AskFunction, dict[str, Any]]:
    match optimizer:
        # Predefined string
        case str():
            return _load_optimizer_from_string(optimizer, space)

        # class/builder
        case Callable():
            info = extract_keyword_defaults(optimizer)
            _optimizer = optimizer(space)
            info["name"] = optimizer.__name__
            return _optimizer, info

        # Predefined string with kwargs
        case (opt, kwargs) if isinstance(opt, str):
            return _load_optimizer_from_string(opt, space, optimizer_kwargs=kwargs)

        # class/builder with kwargs
        case (opt, kwargs):
            info = extract_keyword_defaults(opt)  # type: ignore
            info["name"] = opt.__name__  # type: ignore
            _optimizer = opt(space, **kwargs)  # type: ignore
            return _optimizer, info

        # Mapping with a name
        case {"name": name, **kwargs}:
            return _load_optimizer_from_string(name, space, optimizer_kwargs=kwargs)

        case _:
            raise ValueError(
                f"Unrecognized `optimizer` of type {type(optimizer)}."
                f" {optimizer}. Must either be a string or a callable."
            )
