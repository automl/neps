"""Contains the [`SearchSpace`][neps.space.search_space.SearchSpace] class
which contains the hyperparameters for the search space, as well as
any fidelities and constants.
"""

# mypy: disable-error-code="unreachable"

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from neps.space.neps_spaces.parameters import (
    Categorical as PSCategorical,
    ConfidenceLevel,
    Float as PSFloat,
    Integer as PSInteger,
)
from neps.space.parameters import (
    HPOCategorical,
    HPOConstant,
    HPOFloat,
    HPOInteger,
    Parameter,
)


# NOTE: The use of `Mapping` instead of `dict` is so that type-checkers
# can check if we accidetally mutate these as we pass the parameters around.
# We really should not, and instead make a copy if we really need to.
@dataclass
class SearchSpace(
    Mapping[str, Parameter | HPOConstant | PSCategorical | PSFloat | PSInteger]
):
    """A container for parameters."""

    elements: Mapping[str, Parameter | HPOConstant] = field(default_factory=dict)
    """All items in the search space."""

    categoricals: Mapping[str, HPOCategorical] = field(init=False)
    """The categorical hyperparameters in the search space."""

    numerical: Mapping[str, HPOInteger | HPOFloat] = field(init=False)
    """The numerical hyperparameters in the search space.

    !!! note

        This does not include fidelities.
    """

    fidelities: Mapping[str, HPOInteger | HPOFloat] = field(init=False)
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
    def fidelity(self) -> tuple[str, HPOFloat | HPOInteger] | None:
        """The fidelity parameter for the search space."""
        return None if len(self.fidelities) == 0 else next(iter(self.fidelities.items()))

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        # Convert new PipelineSpace parameters to HPO equivalents if needed
        converted_elements = {}
        for name, hp in self.elements.items():
            # Check if it's a new PipelineSpace parameter
            if isinstance(hp, PSFloat | PSInteger | PSCategorical):
                # The user will get a warning from neps.run about using SearchSpace
                if isinstance(hp, PSFloat):
                    # Extract prior if it exists
                    prior: float | None = None
                    if hp.has_prior and hp.prior is not None:
                        prior = float(hp.prior)
                    # Get string value from ConfidenceLevel enum
                    prior_confidence: str = "low"
                    if hp.has_prior:
                        conf = hp.prior_confidence
                        prior_confidence = (
                            conf.value if isinstance(conf, ConfidenceLevel) else conf
                        )
                    converted_elements[name] = HPOFloat(
                        lower=float(hp.lower),
                        upper=float(hp.upper),
                        log=hp.log,
                        prior=prior,
                        prior_confidence=prior_confidence,  # type: ignore[arg-type]
                        is_fidelity=getattr(hp, "_is_fidelity_compat", False),
                    )
                elif isinstance(hp, PSInteger):
                    # Extract prior if it exists
                    prior_int: int | None = None
                    if hp.has_prior and hp.prior is not None:
                        prior_int = int(hp.prior)
                    # Get string value from ConfidenceLevel enum
                    prior_confidence_int: str = "low"
                    if hp.has_prior:
                        conf = hp.prior_confidence
                        prior_confidence_int = (
                            conf.value if isinstance(conf, ConfidenceLevel) else conf
                        )
                    converted_elements[name] = HPOInteger(
                        lower=int(hp.lower),
                        upper=int(hp.upper),
                        log=hp.log,
                        prior=prior_int,
                        prior_confidence=prior_confidence_int,  # type: ignore[arg-type]
                        is_fidelity=getattr(hp, "_is_fidelity_compat", False),
                        is_scaling=hp.is_scaling,
                    )
                elif isinstance(hp, PSCategorical):
                    # Categorical conversion - extract choices as list
                    # For SearchSpace, choices should be simple list[float | int | str]
                    if isinstance(hp.choices, tuple):
                        choices: list[float | int | str] = list(hp.choices)  # type: ignore[arg-type]
                    else:
                        # If it's a Domain or complex structure, we can't easily convert
                        # Just try to use it as-is and let HPOCategorical validate
                        choices = list(hp.choices)  # type: ignore[arg-type, assignment]

                    # Extract prior if it exists
                    # In PipelineSpace, prior is index; in SearchSpace, actual value
                    prior_cat: float | int | str | None = None
                    if (
                        hp.has_prior
                        and isinstance(hp.prior, int)
                        and 0 <= hp.prior < len(choices)
                    ):
                        # Convert index to actual choice
                        prior_cat = choices[hp.prior]  # type: ignore[assignment]

                    # Get string value from ConfidenceLevel enum
                    prior_confidence_cat: str = "low"
                    if hp.has_prior:
                        conf = hp.prior_confidence
                        prior_confidence_cat = (
                            conf.value if isinstance(conf, ConfidenceLevel) else conf
                        )

                    converted_elements[name] = HPOCategorical(
                        choices=choices,
                        prior=prior_cat,
                        prior_confidence=prior_confidence_cat,  # type: ignore[arg-type]
                    )
            else:
                converted_elements[name] = hp

        # Ensure that we have a consistent order for all our items.
        self.elements = dict(sorted(converted_elements.items(), key=lambda x: x[0]))

        fidelities: dict[str, HPOFloat | HPOInteger] = {}
        numerical: dict[str, HPOFloat | HPOInteger] = {}
        categoricals: dict[str, HPOCategorical] = {}
        arch_params: dict[str, HPOInteger] = {}
        constants: dict[str, Any] = {}

        # Process the hyperparameters
        for name, hp in self.elements.items():
            match hp:
                case HPOFloat() | HPOInteger() if hp.is_fidelity:
                    # We should allow this at some point, but until we do,
                    # raise an error
                    if len(fidelities) >= 1:
                        raise ValueError(
                            "neps only supports one fidelity parameter in the"
                            " pipeline space, but multiple were given."
                            f" Fidelities: {fidelities}, new: {name}"
                        )
                    fidelities[name] = hp
                
                case HPOInteger() if hp.is_scaling:
                    arch_params[name] = hp
                    numerical[name] = hp

                case HPOFloat() | HPOInteger():
                    numerical[name] = hp
                case HPOCategorical():
                    categoricals[name] = hp
                case HPOConstant():
                    constants[name] = hp.value

                case _:
                    raise ValueError(f"Unknown hyperparameter type: {hp}")

        self.categoricals = categoricals
        self.numerical = numerical
        self.constants = constants
        self.fidelities = fidelities
        self.arch_params = arch_params

    def __getitem__(self, key: str) -> Parameter | HPOConstant:
        return self.elements[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)
