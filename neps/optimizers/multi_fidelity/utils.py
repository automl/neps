from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from neps.state.trial import Trial


def trials_to_table(trials: Mapping[str, Trial]) -> pd.DataFrame:
    id_index = np.empty(len(trials), dtype=int)
    rungs_index = np.empty(len(trials), dtype=int)
    perfs = np.empty(len(trials), dtype=np.float64)
    configs = np.empty(len(trials), dtype=object)

    for i, (trial_id, trial) in enumerate(trials.items()):
        config_id_str, rung_str = trial_id.split("_")
        _id, _rung = int(config_id_str), int(rung_str)

        if trial.report is None:
            perf = np.nan  # Pending
        elif trial.report.loss is None:
            perf = np.inf  # Error? Either way, we wont promote it
        else:
            perf = trial.report.loss

        id_index[i] = _id
        rungs_index[i] = _rung
        perfs[i] = perf
        configs[i] = trial.config

    id_index = pd.MultiIndex.from_arrays([id_index, rungs_index], names=["id", "rung"])
    df = pd.DataFrame(data={"config": configs, "perf": perfs}, index=id_index)
    return df.sort_index(ascending=True)
