from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, TypeAlias, TypeVar

import numpy as np
import pandas as pd

from neps.search_spaces.parameter2 import (
    ConstantParameter,
    Parameter,
)
from neps.utils.types import NotSet

if TYPE_CHECKING:
    from numpy.random import Generator

    from neps.search_spaces.distributions import Distribution
    from neps.search_spaces.domain import (
        Domain,
    )
    from neps.utils.types import Array

T = TypeVar("T")
V = TypeVar("V", bound=np.generic)
S = TypeVar("S", bound=np.generic)

Config: TypeAlias = dict[str, Any]
TODOType = Any


class SearchSpace:
    def __init__(
        self,
        _deprecated_fidelity_value: float | None = None,
        **hyperparameters: Parameter | ConstantParameter | int | float | str | bool,
    ):
        _sorted = sorted(hyperparameters.items(), key=lambda x: x[0])

        self.constants: dict[str, int | float | str | bool] = {}
        self.hyperparameters: dict[str, Parameter] = {}
        self._domains: dict[str, Domain] = {}
        self.fidelity_param: tuple[str, Parameter] | None = None
        self.has_prior: bool = False

        for name, hp in _sorted:
            if not isinstance(hp, Parameter):
                self.constants[name] = (
                    hp.value if isinstance(hp, ConstantParameter) else hp
                )
                continue

            if hp.is_fidelity:
                if self.fidelity_param is not None:
                    raise ValueError(
                        "NePS only supports one fidelity parameter in the pipeline space,"
                        " but multiple were given. The previously found fidelity was "
                        f" '{self.fidelity_param[0]}' and the new one is '{name}'"
                    )
                self.fidelity_param = (name, hp)

            if hp.default is not None:
                self.has_prior = True

            self.hyperparameters[name] = hp
            self._domains[name] = hp.domain

        # TODO(eddiebergman): This should be a seperate thing most likely and not
        # in a `SearchSpace`.
        # Variables for tabular bookkeeping
        self.custom_grid_table: pd.Series | pd.DataFrame | None = None
        self.raw_tabular_space: SearchSpace | None = None
        self.has_tabular: bool = False

        # TODO(eddiebergman): This is backwards compatibility progression and should
        # be removed
        self._deprecated_fidelity_value = _deprecated_fidelity_value

    @property
    def deprecated_fidelity_value(self) -> float | None:
        return self._deprecated_fidelity_value

    def deprecated_set_fidelity_value(self, value: float | None) -> None:
        self._deprecated_fidelity_value = value

    def deprecated_clone(self) -> SearchSpace:
        return SearchSpace(
            _deprecated_fidelity_value=self._deprecated_fidelity_value,
            **self.hyperparameters,
        )

    @property
    def fidelity(self) -> Parameter | None:
        return self.fidelity_param[1] if self.fidelity_param is not None else None

    @property
    def fidelity_name(self) -> str | None:
        return self.fidelity_param[0] if self.fidelity_param is not None else None

    # TODO(eddiebergman): Ideally we remove this parameter and leave it to the underlying
    # optimizer to fetch these, disentangling how to sample from the definition of search
    # space
    def uniform_distributions(self) -> dict[str, Distribution]:
        return {k: hp.uniform_distribution() for k, hp in self.hyperparameters.items()}

    def prior_distributions(
        self,
        priors: Mapping[str, tuple[Any, float]],
        *,
        replace_missing_with_uniform: bool = True,
    ) -> dict[str, Distribution]:
        dists: dict[str, Distribution] = {}
        for name, hp in self.hyperparameters.items():
            value, confidence = priors.get(name, (NotSet, 0.0))
            if value is NotSet:
                dists[name] = hp.prior_distribution(value, confidence)
            elif replace_missing_with_uniform:
                dists[name] = hp.uniform_distribution()
            else:
                raise ValueError(
                    f"Missing prior for hyperparameter: '{name}' of type {type(hp)}."
                    " Please provide a prior or set `replace_missing_with_uniform=True`."
                    f"\nReceived priors: {priors}"
                    f"\nHyperparameters: {self.hyperparameters}"
                )
        return dists


def sample_configs(
    searchspace: SearchSpace,
    n: int,
    distributions: Mapping[str, Distribution],
    *,
    seed: Generator,
    ignore_fidelity: bool = False,
    with_constants: bool = True,
) -> list[Config]:
    return to_configs(
        searchspace,
        frame=sample_vectorized(
            searchspace,
            n,
            distributions,
            seed=seed,
            ignore_fidelity=ignore_fidelity,
        ),
        with_constants=with_constants,
    )


def to_configs(
    searchspace: SearchSpace,
    frame: pd.DataFrame,
    *,
    with_constants: bool = True,
) -> list[Config]:
    if with_constants:
        for name, value in searchspace.constants.items():
            if name not in frame:
                frame[name] = value
    else:
        for name in searchspace.constants:
            if name in frame:
                frame = frame.drop(columns=[name], errors="ignore")

    return frame.to_dict(orient="records")


def sample_vectorized(
    searchspace: SearchSpace,
    n: int,
    distributions: Mapping[str, Distribution],
    *,
    seed: Generator,
    ignore_fidelity: bool = False,
) -> pd.DataFrame:
    param_values: dict[str, Array] = {}
    for name, hp in searchspace.hyperparameters.items():
        if hp.is_fidelity and ignore_fidelity:
            continue

        if name not in distributions:
            raise ValueError(
                f"No distribution for hyperparameter: '{name}' of type {type(hp)}"
            )

        distribution = distributions[name]
        samples = distribution.sample(n, seed=seed)
        param_values[name] = hp.domain.cast(samples, frm=distribution.domain)

    # Don't need to copy as we have generated the values and we want to _move_
    # them into here
    return pd.DataFrame(param_values, copy=False)
