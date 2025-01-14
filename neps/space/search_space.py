"""Contains the [`SearchSpace`][neps.search_spaces.search_space.SearchSpace] class
which is a container for hyperparameters that can be sampled, mutated, and crossed over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neps.space.parameters import Categorical, Constant, Float, Integer, Parameter


@dataclass
class SearchSpace:
    """A container for parameters."""

    parameters: dict[str, Parameter]
    """The parameters which define the search space."""

    fidelity: tuple[str, Float | Integer] | None = field(init=False, default=None)
    """The fidelity parameter for the search space."""

    prior: dict[str, Any] = field(init=False, default_factory=dict)
    """The prior configuration for the search space."""

    categoricals: dict[str, Categorical] = field(init=False, default_factory=dict)
    """The categorical hyperparameters in the search space."""

    numerical: dict[str, Integer | Float] = field(init=False, default_factory=dict)
    """The numerical hyperparameters in the search space."""

    constants: dict[str, Any] = field(init=False, default_factory=dict)
    """The constant hyperparameters in the search space."""

    fidelities: dict[str, Integer | Float] = field(init=False, default_factory=dict)
    """The fidelity hyperparameters in the search space.

    Currently no optimizer supports multiple fidelities but it is defined here incase.
    """

    def __post_init__(self) -> None:
        self.parameters = dict(sorted(self.parameters.items(), key=lambda x: x[0]))

        # Process the hyperparameters
        for name, hp in self.parameters.items():
            match hp:
                case Float() | Integer():
                    if hp.is_fidelity:
                        if self.fidelity is not None:
                            raise ValueError(
                                "neps only supports one fidelity parameter in the"
                                " pipeline space, but multiple were given."
                                f" Other fidelity: {self.fidelity[0]}, new: {name}"
                            )
                        self.fidelity = (name, hp)
                        self.fidelities[name] = hp

                    self.numerical[name] = hp

                    if hp.prior is not None:
                        self.prior[name] = hp.prior

                case Categorical():
                    self.categoricals[name] = hp

                    if hp.prior is not None:
                        self.prior[name] = hp.prior

                case Constant():
                    self.constants[name] = hp.value
                    self.prior[name] = hp.value

                case _:
                    raise ValueError(f"Unknown hyperparameter type: {hp}")
