"""Contains the [`SearchSpace`][neps.search_spaces.search_space.SearchSpace] class
which is a container for hyperparameters that can be sampled, mutated, and crossed over.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, TypeAlias

from neps.search_spaces.architecture.graph_grammar import GraphParameter
from neps.search_spaces.domain import UNIT_FLOAT_DOMAIN
from neps.search_spaces.hyperparameters import (
    Categorical,
    Constant,
    Float,
    Integer,
    Numerical,
)

logger = logging.getLogger(__name__)

Parameter: TypeAlias = Float | Integer | Categorical | Constant | GraphParameter
ParameterWithPrior: TypeAlias = Float | Integer | Categorical | GraphParameter


class SearchSpace:
    """A container for hyperparameters that can be sampled, mutated, and crossed over.

    Provides operations for operating on and generating new configurations from the
    hyperparameters.

    !!! note

        The `SearchSpace` class is both the definition of the search space and also
        a configuration at the same time.

        When refering to the `SearchSpace` as a configuration, the documentation will
        refer to it as a `configuration` or `config`. Otherwise, it will be referred to
        as a `search space`.

    !!! note "TODO"

        This documentation is WIP. If you have any questions, please reach out so we can
        know better what to document.
    """

    def __init__(self, **hyperparameters: Parameter):  # noqa: C901, PLR0912
        """Initialize the SearchSpace with hyperparameters.

        Args:
            **hyperparameters: The hyperparameters that define the search space.
        """
        # Ensure a consistent ordering for uses throughout the lib
        _hyperparameters = sorted(hyperparameters.items(), key=lambda x: x[0])
        _fidelity_param: Numerical | None = None
        _fidelity_name: str | None = None
        _has_prior: bool = False

        for name, hp in _hyperparameters:
            match hp:
                case Numerical() if hp.is_fidelity:
                    if _fidelity_param is not None:
                        raise ValueError(
                            "neps only supports one fidelity parameter in the pipeline space,"
                            " but multiple were given. (Hint: check you pipeline space for "
                            "multiple is_fidelity=True)"
                        )

            if not isinstance(hp, Numerical):
                raise ValueError(f"Only float and integer fidelities supported, got {hp}")

            _fidelity_param = hp
            _fidelity_name = name

            if isinstance(hp, ParameterWithPrior) and hp.has_prior:
                _has_prior = True

        self.hyperparameters: dict[str, Parameter] = dict(_hyperparameters)
        self.fidelity: Numerical | None = _fidelity_param
        self.fidelity_name: str | None = _fidelity_name
        self.has_prior: bool = _has_prior

        self.prior_config = {}
        for name, hp in _hyperparameters:
            if hp.prior is not None:
                self.prior_config[name] = hp.prior
                continue

            match hp:
                case Categorical():
                    first_choice = hp.choices[0]
                    self.prior_config[name] = first_choice
                case Integer() | Float():
                    if hp.is_fidelity:
                        self.prior_config[name] = hp.upper
                        continue

                    midpoint = hp.domain.cast_one(0.5, frm=UNIT_FLOAT_DOMAIN)
                    self.prior_config[name] = midpoint
                case Constant():
                    self.prior_config[name] = hp.value
                case GraphParameter():
                    self.prior_config[name] = hp.prior
                case _:
                    raise TypeError(f"Unknown hyperparameter type {hp}")

        self.categoricals: Mapping[str, Categorical] = {
            k: hp for k, hp in _hyperparameters if isinstance(hp, Categorical)
        }
        self.numerical: Mapping[str, Integer | Float] = {
            k: hp
            for k, hp in _hyperparameters
            if isinstance(hp, Integer | Float) and not hp.is_fidelity
        }
        self.graphs: Mapping[str, GraphParameter] = {
            k: hp for k, hp in _hyperparameters if isinstance(hp, GraphParameter)
        }
        self.constants: Mapping[str, Any] = {
            k: hp.value for k, hp in _hyperparameters if isinstance(hp, Constant)
        }
        # NOTE: For future of multiple fidelities
        self.fidelities: Mapping[str, Integer | Float] = {}
        if _fidelity_param is not None and _fidelity_name is not None:
            assert isinstance(_fidelity_param, Integer | Float)
            self.fidelities = {_fidelity_name: _fidelity_param}
