"""Contains the [`SearchSpace`][neps.space.search_space.SearchSpace] class
which contains the hyperparameters for the search space, as well as
any fidelities and constants.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from neps.space.parameters import Categorical, Constant, Float, Integer, Parameter


# NOTE: The use of `Mapping` instead of `dict` is so that type-checkers
# can check if we accidetally mutate these as we pass the parameters around.
# We really should not, and instead make a copy if we really need to.
@dataclass
class SearchSpace(Mapping[str, Parameter | Constant]):
    """A container for parameters."""

    elements: Mapping[str, Parameter | Constant] = field(default_factory=dict)
    """All items in the search space."""

    categoricals: Mapping[str, Categorical] = field(init=False)
    """The categorical hyperparameters in the search space."""

    prior: dict[str, Any] = field(init=False, default_factory=dict)
    """The prior configuration for the search space.

    !!! warning

        This will only contain the priors for the hyperparameters that have been
        specified. If a hyperparameter does not have a prior, it will not be
        included in this dictionary.

        This is to ensure that anything which can deal with partial priors is
        not fooled into thing some default value is a user set prior. For example,
        all [`Prior`][neps.sampling.Prior] classes support partial priors, giving
        a density around a specified prior and a uniform over the rest of the space.

        If you require a fixed configuration that include the full or partial
        prior, you can use the `centers` attribute, as this is gauranteed to
        have a value for every hyperparameter.

        ```python
        # Ensure centers is first, so that any value specified in prior will
        # override it.
        config = {**space.centers, **space.prior}
        ```
    """

    centers: dict[str, Any] = field(init=False, default_factory=dict)
    """The centers of the hyperparameters in the search space.

    Useful to combine with partial priors in the case that a prior is not fully
    specified but you need some fixed, distinct point in the space.
    """

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
    def searchables(self) -> Mapping[str, Parameter]:
        """The hyperparameters that can be searched over.

        !!! note

            This does not include either constants or fidelities.
        """
        return {**self.numerical, **self.categoricals}

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

                case _:
                    raise ValueError(f"Unknown hyperparameter type: {hp}")

        self.categoricals = categoricals
        self.numerical = numerical
        self.constants = constants
        self.fidelities = fidelities

    def __getitem__(self, key: str) -> Parameter | Constant:
        return self.elements[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)
