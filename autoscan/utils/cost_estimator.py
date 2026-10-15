"""Per-configuration cost estimates used in scaling studies.

A [`CostEstimator`][autoscan.utils.cost_estimator.CostEstimator] wraps either a
function `f(**config) -> float` or, for finite spaces, a JSON table of records:

```json
[
  {"config": {"width": 256, "layers": 6}, "cost": 1.2e9},
  {"config": {"width": 512, "layers": 6}, "cost": 4.7e9}
]
```
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostRecord:
    """One record of a JSON cost table: `{"config": {...}, "cost": x}`."""

    config: Mapping[str, Any]
    """The values of the parameters the table is indexed by."""

    cost: float
    """The cost of every configuration with these values."""

    @classmethod
    def from_dict(cls, raw: Any) -> CostRecord:
        """Parse a record as read from JSON, raising `ValueError` if it is malformed."""
        try:
            record = cls(**raw)
        except TypeError as e:
            names = [f.name for f in fields(cls)]
            raise ValueError(f"must be a mapping with exactly {names}, got {raw}") from e
        if not isinstance(record.config, Mapping):
            raise ValueError(f"'config' must be a mapping, got {record.config!r}")
        return cls(config=record.config, cost=float(record.cost))


@dataclass(frozen=True)
class CostEstimator:
    """Maps a configuration to its estimated cost, e.g. FLOPs."""

    fn: Callable[..., float] | None = None
    """The user function, if the estimator was built from one."""

    table: Mapping[tuple[Any, ...], float] | None = None
    """The lookup table, if the estimator was built from a JSON file."""

    table_keys: tuple[str, ...] = ()
    """The parameters a table is indexed by."""

    _accepted: frozenset[str] | None = field(default=None, repr=False)

    @classmethod
    def from_user(
        cls, spec: CostEstimator | Callable[..., float] | str | Path
    ) -> CostEstimator:
        """Build an estimator from what was passed to `autoscan.run()`."""
        if isinstance(spec, CostEstimator):
            return spec
        # If the spec is a string or Path, treat it as a JSON file
        if isinstance(spec, str | Path):
            return cls.from_json(spec)
        if callable(spec):
            return cls.from_callable(spec)
        raise TypeError(
            "cost_estimator must be a callable or the path to a JSON cost table,"
            f" got {type(spec)}."
        )

    @classmethod
    def from_callable(cls, fn: Callable[..., float]) -> CostEstimator:
        """Wrap a function `fn(**config) -> float`.

        A function without `**kwargs` is only passed the parameters it names.
        """
        accepted: frozenset[str] | None = None
        try:
            parameters = inspect.signature(fn).parameters.values()
        except (TypeError, ValueError):
            parameters = None
        if parameters is not None and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters
        ):
            accepted = frozenset(p.name for p in parameters)

        return cls(fn=fn, _accepted=accepted)

    @classmethod
    def from_json(cls, path: str | Path) -> CostEstimator:
        """Load a lookup table from a JSON list of `{"config": {...}, "cost": x}`."""
        path = Path(path)
        records = json.loads(path.read_text())
        if not isinstance(records, list) or len(records) == 0:
            raise ValueError(
                f"Cost table {path} must be a non-empty JSON list of records like"
                ' {"config": {...}, "cost": 1.0}.'
            )

        keys: tuple[str, ...] | None = None
        table: dict[tuple[Any, ...], float] = {}
        for i, raw in enumerate(records):
            try:
                record = CostRecord.from_dict(raw)
            except ValueError as e:
                raise ValueError(f"Record {i} of cost table {path} {e}.") from e
            record_keys = tuple(sorted(record.config))
            if keys is None:
                keys = record_keys
            elif record_keys != keys:
                raise ValueError(
                    f"Record {i} of cost table {path} is indexed by {record_keys},"
                    f" but the previous records are indexed by {keys}."
                )
            table[_table_key(record.config[k] for k in keys)] = record.cost

        assert keys is not None
        return cls(table=table, table_keys=keys)

    def __call__(self, **config: Any) -> float:
        """Estimate the cost of a configuration, given as keyword arguments."""
        if self.fn is not None:
            if self._accepted is not None:
                config = {k: v for k, v in config.items() if k in self._accepted}
            return float(self.fn(**config))

        assert self.table is not None
        try:
            key = _table_key(config[k] for k in self.table_keys)
        except KeyError as e:
            raise ValueError(
                f"The cost table is indexed by {self.table_keys}, but the configuration"
                f" has no value for {e}."
            ) from e
        try:
            return self.table[key]
        except KeyError:
            raise ValueError(
                "The cost table has no entry for"
                f" {dict(zip(self.table_keys, key, strict=True))}. A cost table is"
                " only valid for a finite space and needs one record for every"
                " combination, use a function as `cost_estimator` otherwise."
            ) from None

    def validate(
        self, parameters: Mapping[str, Any], scaling_parameters: Mapping[str, Any]
    ) -> None:
        """Check a table against the search space it is going to be used with.

        Args:
            parameters: Everything a configuration of the space contains.
            scaling_parameters: Those marked with `is_compute_scaling=True`.
        """
        if self.table is None:
            return
        if unknown := sorted(set(self.table_keys) - set(parameters)):
            raise ValueError(
                f"The cost table is indexed by {unknown}, which are not in the"
                f" pipeline space. Available: {sorted(parameters)}."
            )
        if scaling_parameters and set(self.table_keys) != set(scaling_parameters):
            logger.warning(
                "The cost table is indexed by %s, while the parameters marked with"
                " `is_compute_scaling=True` are %s.",
                list(self.table_keys),
                sorted(scaling_parameters),
            )


def _table_key(values: Any) -> tuple[Any, ...]:
    return tuple(values)
