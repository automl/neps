"""Contains the [`SearchSpace`][neps.space.search_space.SearchSpace] class
which contains the hyperparameters for the search space, as well as
any fidelities and constants.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from neps.space.grammar import Grammar
from neps.space.parameters import (
    Categorical,
    Constant,
    Float,
    Integer,
    Parameter,
)


# NOTE: The use of `Mapping` instead of `dict` is so that type-checkers
# can check if we accidetally mutate these as we pass the parameters around.
# We really should not, and instead make a copy if we really need to.
@dataclass
class SearchSpace(Mapping[str, Parameter | Constant]):
    """A container for parameters."""

    elements: Mapping[str, Parameter | Grammar | Constant] = field(default_factory=dict)
    """All items in the search space."""

    categoricals: Mapping[str, Categorical] = field(init=False)
    """The categorical hyperparameters in the search space."""

    grammars: Mapping[str, Grammar] = field(init=False)
    """The grammar parameters of the search space."""

    numerical: Mapping[str, Integer | Float] = field(init=False)
    """The numerical hyperparameters in the search space.

    !!! note

        This does not include fidelities.
    """

    fidelities: Mapping[str, Integer | Float] = field(init=False)
    """The fidelities in the search space.

    Currently no optimizer supports multiple fidelities but it is defined here incase.
    """

    constants: Mapping[str, Any] = field(init=False, default_factory=dict)
    """The constants in the search space."""

    @property
    def grammar(self) -> tuple[str, Grammar] | None:
        """The grammar parameter for the search space if any."""
        return None if len(self.grammars) == 0 else next(iter(self.grammars.items()))

    @property
    def fidelity(self) -> tuple[str, Float | Integer] | None:
        """The fidelity parameter for the search space."""
        return None if len(self.fidelities) == 0 else next(iter(self.fidelities.items()))

    def __post_init__(self) -> None:
        # Ensure that we have a consistent order for all our items.
        self.elements = dict(sorted(self.elements.items(), key=lambda x: x[0]))

        fidelities: dict[str, Float | Integer] = {}
        numerical: dict[str, Float | Integer] = {}
        categoricals: dict[str, Categorical] = {}
        constants: dict[str, Any] = {}
        grammars: dict[str, Grammar] = {}

        # Process the hyperparameters
        for name, hp in self.elements.items():
            match hp:
                case Float() | Integer() if hp.is_fidelity:
                    # We should allow this at some point, but until we do,
                    # raise an error
                    if len(fidelities) >= 1:
                        raise ValueError(
                            "neps only supports one fidelity parameter in the"
                            " pipeline space, but multiple were given."
                            f" Fidelities: {fidelities}, new: {name}"
                        )
                    fidelities[name] = hp

                case Float() | Integer():
                    numerical[name] = hp
                case Categorical():
                    categoricals[name] = hp
                case Constant():
                    constants[name] = hp.value
                case Grammar():
                    if len(grammars) >= 1:
                        raise ValueError(
                            "neps only supports one grammar parameter in the"
                            " pipeline space, but multiple were given."
                            f" Grammars: {grammars}, new: {name}"
                        )
                    grammars[name] = hp
                case _:
                    raise ValueError(f"Unknown hyperparameter type: {hp}")

        self.categoricals = categoricals
        self.numerical = numerical
        self.constants = constants
        self.fidelities = fidelities
        self.grammars = grammars

    def __getitem__(self, key: str) -> Parameter | Constant | Grammar:
        return self.elements[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)
