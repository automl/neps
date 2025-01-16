"""The selection of optimization algorithms available in NePS.

This module conveniently starts with 'a' to be at the top and
is where most of the code documentation for optimizers can be found.

Below you will find some functions with some sane defaults documenting
the parameters available. You can pass these functoins to `neps.run()`
if you like, otherwise you may also refer to them by their string name.
"""
# NOTE: If updating this file with new optimizers, please be aware that
# the documentation here is what is shown in the `neps.run()` documentation.
# Heres a checklist:
# 1. Add you function and document it
# 2. Add it to the `OptimizerChoice` at the bottom of this file.
# 3. Add a section to `neps.run()`

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Concatenate, Literal, TypeAlias

import torch

from neps.optimizers.ask_and_tell import AskAndTell  # noqa: F401
from neps.optimizers.bayesian_optimization import BayesianOptimization
from neps.optimizers.bracket_optimizer import BracketOptimizer
from neps.optimizers.grid_search import GridSearch
from neps.optimizers.ifbo import IFBO
from neps.optimizers.models.ftpfn import FTPFNSurrogate
from neps.optimizers.optimizer import AskFunction  # noqa: TC001
from neps.optimizers.priorband import PriorBandArgs
from neps.optimizers.random_search import RandomSearch
from neps.sampling import Prior, Sampler, Uniform
from neps.space.encoding import CategoricalToUnitNorm, ConfigEncoder

if TYPE_CHECKING:
    import pandas as pd

    from neps.optimizers.utils.brackets import Bracket
    from neps.space import SearchSpace


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
    sampler: Literal["uniform", "prior", "priorband"] | PriorBandArgs | Sampler,
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

            * If a `PriorBandArgs` object, samples with weights according to the
                PriorBand algorithm with the given parameters.
            * If a `Sampler` object, samples from the space using the sampler.

        sample_prior_first: Whether to sample the prior configuration first.
    """
    assert pipeline_space.fidelity is not None
    fidelity_name, fidelity = pipeline_space.fidelity

    if len(pipeline_space.fidelities) != 1:
        raise ValueError(
            "Only one fidelity should be defined in the pipeline space."
            f"\nGot: {pipeline_space.fidelities}"
        )

    if sample_prior_first not in (True, False, "highest_fidelity"):
        raise ValueError(
            "sample_prior_first should be either True, False or 'highest_fidelity'"
        )

    from neps.optimizers.utils import brackets

    # Determine the strategy for creating brackets for sampling
    create_brackets: Callable[[pd.DataFrame], Sequence[Bracket] | Bracket]
    match bracket_type:
        case "successive_halving":
            assert early_stopping_rate is not None
            rung_to_fidelity, rung_sizes = brackets.calculate_sh_rungs(
                bounds=(fidelity.lower, fidelity.upper),
                eta=eta,
                early_stopping_rate=early_stopping_rate,
            )
            create_brackets = partial(
                brackets.Sync.create_repeating, rung_sizes=rung_sizes
            )
        case "hyperband":
            assert early_stopping_rate is None
            rung_to_fidelity, bracket_layouts = brackets.calculate_hb_bracket_layouts(
                bounds=(fidelity.lower, fidelity.upper),
                eta=eta,
            )
            create_brackets = partial(
                brackets.Hyperband.create_repeating,
                bracket_layouts=bracket_layouts,
            )
        case "asha":
            assert early_stopping_rate is not None
            rung_to_fidelity, _rung_sizes = brackets.calculate_sh_rungs(
                bounds=(fidelity.lower, fidelity.upper),
                eta=eta,
                early_stopping_rate=early_stopping_rate,
            )
            create_brackets = partial(
                brackets.Async.create, rungs=list(rung_to_fidelity), eta=eta
            )
        case "async_hb":
            assert early_stopping_rate is None
            rung_to_fidelity, bracket_layouts = brackets.calculate_hb_bracket_layouts(
                bounds=(fidelity.lower, fidelity.upper),
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

    _sampler: Sampler | PriorBandArgs
    match sampler:
        case "uniform":
            _sampler = Sampler.uniform(ndim=encoder.ndim)
        case "prior":
            _sampler = Prior.from_config(pipeline_space.prior, space=pipeline_space)
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
        fid_min=fidelity.lower,
        fid_max=fidelity.upper,
        fid_name=fidelity_name,
        sampler=_sampler,
        sample_prior_first=sample_prior_first,
        create_brackets=create_brackets,
    )


def determine_optimizer_automatically(space: SearchSpace) -> str:
    if len(space.prior) > 0:
        return "priorband" if len(space.fidelities) > 0 else "pibo"

    return "hyperband" if len(space.fidelities) > 0 else "bayesian_optimization"


def random_search(
    pipeline_space: SearchSpace,
    *,
    use_priors: bool = False,
    ignore_fidelity: bool = True,
) -> RandomSearch:
    """A simple random search algorithm that samples configurations uniformly at random.

    You may also `use_priors=` to sample from a distribution centered around your defined
    priors.

    Args:
        pipeline_space: The search space to sample from.
        use_priors: Whether to use priors when sampling.
        ignore_fidelity: Whether to ignore fidelity when sampling.
            In this case, the max fidelity is always used.
    """
    encoder = ConfigEncoder.from_space(
        pipeline_space, include_fidelity=not ignore_fidelity
    )

    sampler: Sampler
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


def grid_search(pipeline_space: SearchSpace) -> GridSearch:
    """A simple grid search algorithm which discretizes the search
    space and evaluates all possible configurations.

    Args:
        pipeline_space: The search space to sample from.
    """
    from neps.optimizers.utils.grid import make_grid

    return GridSearch(
        pipeline_space=pipeline_space,
        configs_list=make_grid(pipeline_space),
    )


def ifbo(
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
    """A transformer that has been trained to predict loss curves of deep-learing
    models, used to guide the optimization procedure and select configurations which
    are most promising to evaluate.

    !!! tip "When to use this?"

        Use this when you think that early signal in your loss curve could be used
        to distinguish which configurations are likely to achieve a good performance.

        This algorithm will take many small steps in evaluating your configuration
        so we also advise that saving and loading your model checkpoint should
        be relatively fast.

    This algorithm requires a _fidelity_ parameter, such as `epochs`, to be present.
    Each time we evaluate a configuration, we will only evaluate it for a single
    epoch, before returning back to the ifbo algorithm to select the next configuration.

    ??? tip "Fidelities?"

        A fidelity parameter lets you control how many resources to invest in
        a single evaluation. For example, a common one for deep-learing is
        `epochs`. We can evaluate a model for just a single epoch, (fidelity step)
        to gain more information about the model's performance and decide what
        to do next.

    * **Paper**: https://openreview.net/forum?id=VyoY3Wh9Wd
    * **Github**: https://github.com/automl/ifBO/tree/main

    Args:
        pipeline_space: Space in which to search
        step_size: The size of the step to take in the fidelity domain.
        sample_prior_first: Whether to sample the default configuration first
        initial_design_size: Number of configs to sample before starting optimization

            If `None`, the number of configs will be equal to the number of dimensions.

        device: Device to use for the model
        surrogate_path: Path to the surrogate model to use
        surrogate_version: Version of the surrogate model to use
    """
    from neps.optimizers.ifbo import _adjust_space_to_match_stepsize

    # TODO: I'm not sure how this might effect tables, whose lowest fidelity
    # might be below to possibly increased lower bound.
    space, fid_bins = _adjust_space_to_match_stepsize(pipeline_space, step_size)
    assert space.fidelity is not None
    fidelity_name, fidelity = space.fidelity

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
        fid_domain=fidelity.domain,
        fidelity_name=fidelity_name,
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


def successive_halving(
    space: SearchSpace,
    *,
    sampler: Literal["uniform", "prior"] = "uniform",
    eta: int = 3,
    early_stopping_rate: int = 0,
    sample_prior_first: bool | Literal["highest_fidelity"] = False,
) -> BracketOptimizer:
    """
    A bandit-based optimization algorithm that uses a _fidelity_ parameter
    to gradually invest resources into more promising configurations.

    ??? tip "Fidelities?"

        A fidelity parameter lets you control how many resources to invest in
        a single evaluation. For example, a common one for deep-learing is
        `epochs`. By evaluating a model for just a few epochs, we can quickly
        get a sense if the model is promising or not. Only those that perform
        well get _promoted_ and evaluated at a higher epoch.

    !!! tip "When to use this?"

        When you think that the rank of N configurations at a lower fidelity correlates
        very well with the rank if you were to evaluate those configurations at higher
        fidelities.

    It does this by creating a competition between N configurations and
    racing them in a _bracket_ against each other.
    This _bracket_ has a series of incrementing _rungs_, where lower rungs
    indicate less resources invested. The amount of resources is related
    to your fidelity parameter, with the highest rung relating to the
    maximum of your fidelity parameter.

    Those that perform well get _promoted_ and evaluated with more resources.

    ```
    # A bracket indicating the rungs and configurations.
    # Those which performed best get promoted through the rungs.

    |        | fidelity    | c1 | c2 | c3 | c4 | c5 | ... | cN |
    | Rung 0 | (3 epochs)  |  o |  o |  o |  o |  o | ... | o  |
    | Rung 1 | (9 epochs)  |  o |    |  o |  o |    | ... | o  |
    | Rung 2 | (27 epochs) |  o |    |    |    |    | ... |    |
    ```

    By default, new configurations are sampled using a _uniform_ distribution,
    however you can also specify to prefer sampling from around a distribution you
    think is more promising by setting the `prior` and the `prior_confidence`
    of parameters of your search space.

    You can choose between these by setting `#!python sampler="uniform"`
    or `#!python sampler="prior"`.

    Args:
        space: The search space to sample from.
        eta: The reduction factor used for building brackets
        early_stopping_rate: Determines the number of rungs in a bracket
            Choosing 0 creates maximal rungs given the fidelity bounds.
        sampler: The type of sampling procedure to use:

            * If `#!python "uniform"`, samples uniformly from the space when
                it needs to sample.
            * If `#!python "prior"`, samples from the prior
                distribution built from the `prior` and `prior_confidence`
                values in the search space.

        sample_prior_first: Whether to sample the prior configuration first,
            and if so, should it be at the highest fidelity level.
    """
    return _bracket_optimizer(
        pipeline_space=space,
        bracket_type="successive_halving",
        eta=eta,
        early_stopping_rate=early_stopping_rate,
        sampler=sampler,
        sample_prior_first=sample_prior_first,
    )


def hyperband(
    space: SearchSpace,
    *,
    eta: int = 3,
    sampler: Literal["uniform", "prior"] = "uniform",
    sample_prior_first: bool | Literal["highest_fidelity"] = False,
) -> BracketOptimizer:
    """Another bandit-based optimization algorithm that uses a _fidelity_ parameter,
    very similar to [`successive_halving`][neps.optimizers.algorithms.successive_halving],
    but hedges a bit more on the safe side, just incase your _fidelity_ parameters
    isn't as well correlated as you'd like.

    !!! tip "When to use this?"

        Use this when you think lower fidelity evaluations of your configurations carries
        some signal about their ranking at higher fidelities, but not enough to be certain

    Hyperband is like Successive Halving but it instead of always having the same bracket
    layout, it runs different brackets with different rungs.

    This helps hedge against scenarios where rankings at the lowest fidelity do
    not correlate well with the upper fidelity.


    ```
    # Hyperband runs different successive halving brackets

    | Bracket 1 |         | Bracket 2 |        | Bracket 3 |
    | Rung 0    | ... |   | (skipped) |        | (skipped) |
    | Rung 1    | ... |   | Rung 1    | ... |  | (skipped) |
    | Rung 2    | ... |   | Rung 2    | ... |  | Rung 2    | ... |
    ```

    For more information, see the
    [`successive_halving`][neps.optimizers.algorithms.successive_halving] documentation,
    as this algorithm could be considered an extension of it.

    Args:
        space: The search space to sample from.
        eta: The reduction factor used for building brackets
        sampler: The type of sampling procedure to use:

            * If `#!python "uniform"`, samples uniformly from the space when
                it needs to sample.
            * If `#!python "prior"`, samples from the prior
                distribution built from the `prior` and `prior_confidence`
                values in the search space.

        sample_prior_first: Whether to sample the prior configuration first,
            and if so, should it be at the highest fidelity level.
    """
    return _bracket_optimizer(
        pipeline_space=space,
        bracket_type="hyperband",
        eta=eta,
        sampler=sampler,
        sample_prior_first=sample_prior_first,
    )


def asha(
    space: SearchSpace,
    *,
    eta: int = 3,
    early_stopping_rate: int = 0,
    sampler: Literal["uniform", "prior"] = "uniform",
    sample_prior_first: bool | Literal["highest_fidelity"] = False,
) -> BracketOptimizer:
    """A bandit-based optimization algorithm that uses a _fidelity_ parameter,
    the _asynchronous_ version of
    [`successive_halving`][neps.optimizers.algorithms.successive_halving].
    one that scales better to many parallel workers.

    !!! tip "When to use this?"

        Use this when you think lower fidelity evaluations of your configurations carries
        a strong signal about their ranking at higher fidelities, and you have many
        workers available to evaluate configurations in parallel.

    It does this by maintaining one big bracket, i.e. one
    big on-going competition, with a promotion rule based on the sizes of each rung.

    ```
    # ASHA maintains one big bracket with an exponentially decreasing amount of
    # configurations promoted, relative to those in the rung below.

    |        | fidelity    | c1 | c2 | c3 | c4 | c5 | ...
    | Rung 0 | (3 epochs)  |  o |  o |  o |  o |  o | ...
    | Rung 1 | (9 epochs)  |  o |    |  o |  o |    | ...
    | Rung 2 | (27 epochs) |  o |    |    |  o |    | ...
    ```

    For more information, see the
    [`successive_halving`][neps.optimizers.algorithms.successive_halving] documentation,
    as this algorithm could be considered an extension of it.

    Args:
        space: The search space to sample from.
        eta: The reduction factor used for building brackets
        sampler: The type of sampling procedure to use:

            * If `#!python "uniform"`, samples uniformly from the space when
                it needs to sample.
            * If `#!python "prior"`, samples from the prior
                distribution built from the `prior` and `prior_confidence`
                values in the search space.

        sample_prior_first: Whether to sample the prior configuration first,
            and if so, should it be at the highest fidelity.
    """
    return _bracket_optimizer(
        pipeline_space=space,
        bracket_type="asha",
        eta=eta,
        early_stopping_rate=early_stopping_rate,
        sampler=sampler,
        sample_prior_first=sample_prior_first,
    )


def async_hb(
    space: SearchSpace,
    *,
    eta: int = 3,
    sampler: Literal["uniform", "prior"] = "uniform",
    sample_prior_first: bool = False,
) -> BracketOptimizer:
    """An _asynchronous_ version of [`hyperband`][neps.optimizers.algorithms.hyperband],
    where the brackets are run asynchronously, and the promotion rule is based on the
    number of evaluations each configuration has had.

    !!! tip "When to use this?"

        Use this when you think lower fidelity evaluations of your configurations carries
        some signal about their ranking at higher fidelities, but not confidently, and
        you have many workers available to evaluate configurations in parallel.

    ```
    # Async HB runs different "asha" brackets, which are unbounded in the number
    # of configurations that can be in each. The bracket chosen at each iteration
    # is a sampling function based on the resources invested in each bracket.

    | Bracket 1 |         | Bracket 2 |        | Bracket 3 |
    | Rung 0    | ...     | (skipped) |        | (skipped) |
    | Rung 1    | ...     | Rung 1    | ...    | (skipped) |
    | Rung 2    | ...     | Rung 2    | ...    | Rung 2    | ...
    ```

    For more information, see the
    [`hyperband`][neps.optimizers.algorithms.hyperband] documentation,
    [`successive_halving`][neps.optimizers.algorithms.successive_halving] documentation,
    and the [`asha`][neps.optimizers.algorithms.asha] documentation, as this algorithm
    takes elements from each.

    Args:
        space: The search space to sample from.
        eta: The reduction factor used for building brackets
        sampler: The type of sampling procedure to use:

            * If `#!python "uniform"`, samples uniformly from the space when
                it needs to sample.
            * If `#!python "prior"`, samples from the prior
                distribution built from the `prior` and `prior_confidence`
                values in the search space.

        sample_prior_first: Whether to sample the prior configuration first.
    """
    return _bracket_optimizer(
        pipeline_space=space,
        bracket_type="async_hb",
        eta=eta,
        sampler=sampler,
        sample_prior_first=sample_prior_first,
    )


def priorband(
    space: SearchSpace,
    *,
    eta: int = 3,
    sample_prior_first: bool | Literal["highest_fidelity"] = False,
    base: Literal["successive_halving", "hyperband", "asha", "async_hb"] = "hyperband",
) -> BracketOptimizer:
    """Priorband is also a bandit-based optimization algorithm that uses a _fidelity_,
    providing a general purpose sampling extension to other algorithms. It makes better
    use of the prior information you provide in the search space along with the fact
    that you can afford to explore and take more risk at lower fidelities.

    !!! tip "When to use this?"

        Use this when you have a good idea of what good parameters look like and
        can specify them through the `prior` and `prior_confidence` parameters in
        the search space.

        As `priorband` is flexible, you may choose between the existing tradeoffs
        the other algorithms provide through the use of `base=`.

    Priorband works by adjusting the sampling procedure to sample from one of
    the following three distributions:

    * 1) a uniform distribution
    * 2) a prior distribution
    * 3) a distribution around the best found configuration so far.

    By weighing the likelihood of good configurations having been sampled
    from each of these distribution, we can score them against each other to aid
    selection. We further use the fact that we can afford to explore and take more
    risk at lower fidelities, which is factored into the sampling procedure.

    See: https://openreview.net/forum?id=uoiwugtpCH&noteId=xECpK2WH6k

    Args:
        space: The search space to sample from.
        eta: The reduction factor used for building brackets
        sample_prior_first: Whether to sample the prior configuration first.
        base: The base algorithm to use for the bracketing.
    """
    return _bracket_optimizer(
        pipeline_space=space,
        bracket_type=base,
        eta=eta,
        sampler="priorband",
        sample_prior_first=sample_prior_first,
    )


def bayesian_optimization(
    space: SearchSpace,
    *,
    initial_design_size: int | Literal["ndim"] = "ndim",
    cost_aware: bool | Literal["log"] = False,
    device: torch.device | str | None = None,
) -> BayesianOptimization:
    """Models the relation between hyperparameters in your `pipeline_space`
    and the results of `evaluate_pipeline` using bayesian optimization.
    This acts as a cheap _surrogate model_ of you `evaluate_pipeline` function
    that can be used for optimization.

    !!! tip "When to use this?"

        Bayesion optimization is a good general purpose choice, especially
        if the size of your search space is not too large. It is also the best
        option to use if you do not have or want to use a _fidelity_ parameter.

        Note that acquiring the next configuration to evaluate with bayesian
        optimization can become prohibitvely expensive as the number of
        configurations evaluated increases.

    If there is some numeric cost associated with evaluating a configuration,
    you can provide this as a `cost` when returning the results from your
    `evaluate_pipeline` function. By specifying `#!python cost_aware=True`,
    the optimizer will attempt to balance getting the best result while
    minimizing the cost.

    If you have _priors_, we recommend looking at
    [`pibo`][neps.optimizers.algorithms.pibo].

    Args:
        space: The search space to sample from.
        initial_design_size: Number of samples used before using the surrogate model.
            If "ndim", it will use the number of parameters in the search space.
        cost_aware: Whether to consider reported "cost" from configurations in decision
            making. If True, the optimizer will weigh potential candidates by how much
            they cost, incentivising the optimizer to explore cheap, good performing
            configurations. This amount is modified over time. If "log", the cost
            will be log-transformed before being used.

            !!! warning

                If using `cost`, cost must be provided in the reports of the trials.

        device: Device to use for the optimization.
    """
    return _bo(
        pipeline_space=space,
        initial_design_size=initial_design_size,
        cost_aware=cost_aware,
        device=device,
        use_priors=False,
        sample_prior_first=False,
    )


def pibo(
    space: SearchSpace,
    *,
    initial_design_size: int | Literal["ndim"] = "ndim",
    cost_aware: bool | Literal["log"] = False,
    device: torch.device | str | None = None,
    sample_prior_first: bool = False,
) -> BayesianOptimization:
    """A modification of
    [`bayesian_optimization`][neps.optimizers.algorithms.bayesian_optimization]
    that also incorporates the use of priors in the search space.

    !!! tip "When to use this?"

        Use this if you'd like to use bayesian optimization while also having
        a good idea of what good parameters look like and can specify them
        through the `prior` and `prior_confidence` parameters in the search space.

        Note that this incurs the same tradeoffs that bayesian optimization
        has.

    Args:
        space: The search space to sample from.
        initial_design_size: Number of samples used before using the surrogate model.
            If "ndim", it will use the number of parameters in the search space.
        cost_aware: Whether to consider reported "cost" from configurations in decision
            making. If True, the optimizer will weigh potential candidates by how much
            they cost, incentivising the optimizer to explore cheap, good performing
            configurations. This amount is modified over time. If "log", the cost
            will be log-transformed before being used.

            !!! warning

                If using `cost`, cost must be provided in the reports of the trials.

        device: Device to use for the optimization.
    """
    return _bo(
        pipeline_space=space,
        initial_design_size=initial_design_size,
        cost_aware=cost_aware,
        device=device,
        use_priors=True,
        sample_prior_first=sample_prior_first,
    )


PredefinedOptimizers: Mapping[
    str,
    Callable[Concatenate[SearchSpace, ...], AskFunction],
] = {
    f.__name__: f
    for f in (
        bayesian_optimization,
        pibo,
        random_search,
        grid_search,
        ifbo,
        successive_halving,
        hyperband,
        asha,
        async_hb,
        priorband,
    )
}

OptimizerChoice: TypeAlias = Literal[
    "bayesian_optimization",
    "pibo",
    "successive_halving",
    "hyperband",
    "asha",
    "async_hb",
    "priorband",
    "random_search",
    "grid_search",
    "ifbo",
]
