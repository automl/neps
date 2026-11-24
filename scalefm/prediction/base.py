"""What a predictor answers at a compute budget, and which predictor answers what.

The fits live in one module per predictor (`power_law`, `chinchilla3`, `gp`); the
per-trial-prefix trajectories and their caching live in `trajectory`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

Row = tuple[float, float, float, float]
"""One evaluated run as (n_params, n_data, loss, flops)."""


class Target(Enum):
    """What a prediction at a compute budget contains.

    * `LOSS`: the best loss reachable with that compute.
    * `COARSE`: the compute optimal model size and data, `(n_params, n_data)`.
    * `FULL`: the compute optimal full configuration.
    """

    LOSS = "loss"
    COARSE = "coarse"
    FULL = "full"


class PredictorTypes(Enum):
    """`value` is the on-disk slug (cache filenames, result dirs). `display_label` is
    what shows up in figures and tables.
    """

    color: str
    display_label: str
    provides: frozenset[Target]

    def __new__(
        cls, slug: str, color: str, display_label: str, provides: frozenset[Target]
    ) -> PredictorTypes:
        """A member with value `slug` and the given attributes."""
        obj = object.__new__(cls)
        obj._value_ = slug
        obj.color = color
        obj.display_label = display_label
        obj.provides = provides
        return obj

    POWER_LAW = (
        "power_law",
        "#1f77b4",
        "Power Law",
        frozenset({Target.LOSS, Target.COARSE, Target.FULL}),
    )
    CHINCHILLA3 = (
        "chinchilla3",
        "#ff7f0e",
        "Chinchilla3",
        frozenset({Target.LOSS, Target.COARSE}),
    )
    GP_POSTERIOR = (
        "gp_posterior",
        "#2ca02c",
        "GP Posterior",
        frozenset({Target.LOSS, Target.COARSE, Target.FULL}),
    )

    @classmethod
    def providing(cls, target: Target) -> list[PredictorTypes]:
        """The predictors that can predict `target`."""
        return [p for p in cls if target in p.provides]


def check_provides(predictor_type: PredictorTypes, target: Target) -> None:
    """Raise a `ValueError` if `predictor_type` cannot predict `target`."""
    if target not in predictor_type.provides:
        raise ValueError(
            f"{predictor_type.display_label} does not predict {target.value}, "
            f"only {sorted(t.value for t in predictor_type.provides)}."
        )


def flatten_rows(rows: list[Row] | list[list[Row]]) -> list[Row]:
    """Every row, with grouped learning-curve rows concatenated."""
    if not rows:
        return []

    first = rows[0]
    if isinstance(first, tuple):
        return list(rows)  # type: ignore[arg-type]

    # Typed loosely, since the rows are not validated and may be malformed.
    curves: list[Any] = list(rows)
    flattened: list[Row] = []
    for curve in curves:
        if not isinstance(curve, (list, tuple)):
            continue
        for point in curve:
            if isinstance(point, tuple) and len(point) == 4:
                flattened.append(point)
    return flattened


def last_point_in_curve(rows: list[Row] | list[list[Row]]) -> list[Row]:
    """The final row of each grouped learning curve; plain rows are returned as is."""
    if not rows:
        return []

    first = rows[0]
    if isinstance(first, tuple):
        return list(rows)  # type: ignore[arg-type]

    # Typed loosely, since the rows are not validated and may be malformed.
    curves: list[Any] = list(rows)
    flattened: list[Row] = []
    for curve in curves:
        if not isinstance(curve, (list, tuple)):
            continue
        if curve and isinstance(curve[-1], tuple) and len(curve[-1]) == 4:
            flattened.append(curve[-1])
        elif isinstance(curve, tuple) and len(curve) == 4:
            flattened.append(curve)
    return flattened
