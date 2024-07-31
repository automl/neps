from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from neps.runtime import (
    Report,
    StatePaths,
    Trial,
    _sample_trial_from_optimizer
)
from neps.utils.types import TrialID

if TYPE_CHECKING:
    from neps.optimizers.base_optimizer import BaseOptimizer


@dataclass
class AskAndTellWrapper:
    working_dir: Path
    optimizer: BaseOptimizer

    evaluated_trials: dict[TrialID, Report] = field(default_factory=dict)
    pending_trials: dict[TrialID, Trial] = field(default_factory=dict)

    def __post_init__(self):
        self.paths = StatePaths(self.working_dir, create_dirs=True)

    def ask(self) -> Trial:
        # `load_results` and `get_config_and_id` are inside `_sample_trial_from_optimizer`
        # which the `runtime/metahyper` also use.
        trial = _sample_trial_from_optimizer(
            self.optimizer,
            config_dir_f=self.paths.config_dir,
            evaluated_trials=self.evaluated_trials,
            pending_trials=self.pending_trials,
        )
        self.pending_trials[trial.id] = trial
        return trial

    def tell(self, report: Report) -> None:
        if report.cost is not None:
            self.optimizer.used_budget += report.cost

        self.pending_trials.pop(report.id, None)
        self.evaluated_trials[report.id] = report

    def is_out_of_budget(self) -> bool:
        return self.optimizer.is_out_of_budget()
