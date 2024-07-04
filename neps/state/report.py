"""A trial is a configuration and it's associated data."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import numpy as np

from neps.utils.types import ConfigResult, RawConfig

if TYPE_CHECKING:
    from neps.search_spaces import SearchSpace
    from neps.state.trial import Trial, TrialID
    from neps.utils.types import ERROR


logger = logging.getLogger(__name__)


class DeserializedError(Exception):
    """An exception that was deserialized from a file."""


def to_config_result(
    trial: Trial,
    report: Report,
    config_to_search_space: Callable[[RawConfig], SearchSpace],
) -> ConfigResult:
    """Convert the trial and report to a `ConfigResult` object."""
    if report.reported_as == "success":
        result = {
            **report.extra,
            "loss": report.loss,
            "cost": report.cost,
        }
    else:
        result = "error"

    return ConfigResult(
        trial.id,
        config=config_to_search_space(trial.config),
        result=result,
        metadata=asdict(trial.metadata),
    )


@dataclass(kw_only=True)
class Report:
    """A failed report of the evaluation of a configuration."""

    trial_id: TrialID
    loss: float | None
    cost: float | None
    account_for_cost: bool
    extra: Mapping[str, Any]
    err: Exception | None
    tb: str | None
    reported_as: Literal["success", "failed", "crashed"]

    def to_deprecate_result_dict(self) -> dict[str, Any] | ERROR:
        """Return the report as a dictionary."""
        if self.reported_as == "success":
            return {
                "loss": self.loss,
                "cost": self.cost,
                "account_for_cost": self.account_for_cost,
                **self.extra,
            }
        return "error"

    def __eq__(self, value: Any, /) -> bool:  # noqa: C901
        # HACK : Since it could be probably that one of loss or cost is nan,
        # we need a custom comparator for this object
        # HACK : We also have to skip over the `Err` object since when it's deserialized,
        # we can not recover the original object/type.
        if not isinstance(value, Report):
            return False

        other_items = value.__dict__
        for k, v in self.__dict__.items():
            other_v = other_items[k]
            if k == "err" and isinstance(v, tuple) and isinstance(other_v, tuple):
                if isinstance(v[0], DeserializedError) ^ isinstance(
                    other_v[0], DeserializedError
                ):
                    continue

                if v != other_v:
                    return False
            elif k in ("loss", "cost"):
                if v is not None and np.isnan(v):
                    if other_v is None or not np.isnan(other_v):
                        return False
                elif v != other_v:
                    return False
            elif v != other_v:
                return False

        return True
