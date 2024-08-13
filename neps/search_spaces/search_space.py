from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Mapping, TypeAlias, TypeVar

import numpy as np

from neps.search_spaces.architecture import GraphParameter
from neps.search_spaces.config import Config
from neps.search_spaces.hyperparameters import (
    CategoricalParameter,
    ConstantParameter,
    FloatParameter,
    IntegerParameter,
    NumericalParameter,
    Parameter,
)

if TYPE_CHECKING:
    import pandas as pd


T = TypeVar("T")
V = TypeVar("V", bound=np.generic)
S = TypeVar("S", bound=np.generic)

# Should probably replace this with an enum
Confidence: TypeAlias = Literal["low", "medium", "high", "ultra"]


class SearchSpace:
    # (0, 1) Higher is more confident. Details of how this is used
    # is up to the consumer, for example, in a trucnorm, this is
    # used as 1 - confidence to get the standard deviation of the unit
    # norm. For categorical prior sampling, this is the weight of sampling
    # the default value.
    DEFAULT_CONFIDENCE_SCORES: ClassVar[Mapping[Confidence, float]] = {
        "low": 0.5,
        "medium": 0.75,
        "high": 0.875,
        "ultra": 0.95,
    }

    numericals: Mapping[str, NumericalParameter]
    categoricals: Mapping[str, CategoricalParameter]
    graphs: Mapping[str, GraphParameter]

    fidelity: Mapping[str, NumericalParameter]

    constants: Mapping[str, int | float | str | bool]

    initial_prior: dict[str, tuple[Any, float]] | None
    default_configuration: Config
    default_confidence_scores: dict[str, tuple[Any, float]]

    # To deprecate
    has_tabular: bool
    custom_grid_table: pd.Series | pd.DataFrame | None
    raw_tabular_space: SearchSpace | None

    def __init__(self, **hyperparameters: Parameter | GraphParameter):
        _sorted = sorted(hyperparameters.items(), key=lambda x: x[0])
        self.numericals: Mapping[str, NumericalParameter] = {}
        self.categoricals: Mapping[str, CategoricalParameter] = {}
        self.graphs: Mapping[str, GraphParameter] = {}
        self.fidelity: Mapping[str, NumericalParameter] = {}
        self.constants: Mapping[str, int | float | str | bool] = {}

        prior: dict[str, tuple[Any, float]] = {}
        _default: dict[str, Any] = {}
        for name, hp in _sorted:
            if isinstance(hp, ConstantParameter):
                self.constants[name] = hp.value
                _default[name] = hp.value
                continue

            if isinstance(hp, (IntegerParameter, FloatParameter)):
                if hp.is_fidelity:
                    self.fidelity[name] = hp
                else:
                    self.numericals[name] = hp

                if hp.default is not None:
                    prior[name] = (
                        hp.default,
                        self.DEFAULT_CONFIDENCE_SCORES[hp.default_confidence],
                    )
                    _default[name] = hp.default
                else:
                    _default[name] = hp.domain.midpoint
                continue

            if isinstance(hp, CategoricalParameter):
                self.categoricals[name] = hp
                if hp.default is not None:
                    prior[name] = (
                        hp.default,
                        self.DEFAULT_CONFIDENCE_SCORES[hp.default_confidence],
                    )
                    _default[name] = hp.default
                else:
                    _default[name] = hp.choices[0]
                continue

            if isinstance(hp, GraphParameter):
                self.graphs[name] = hp
                # TODO: What is the midpoint of a graph?? I don't like that
                # this has to undertministacally sample it.
                _default[name] = hp.sample()
                continue

            raise NotImplementedError(f"Unknown Parameter type: {type(hp)}\n{hp}")

        if self.fidelity:
            max_fidelities = {k: v.domain.upper for k, v in self.fidelity.items()}
        else:
            max_fidelities = None

        self.initial_prior = prior if len(prior) > 0 else None
        self.default_configuration = Config.new(_default, fidelity=max_fidelities)
        self.hyperparameters = {**self.numericals, **self.categoricals, **self.graphs}

    @property
    def has_prior(self) -> bool:
        return self.initial_prior is not None
