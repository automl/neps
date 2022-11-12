from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from typing_extensions import Literal

from metahyper.api import ConfigResult

from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.hyperband import HyperbandCustomDefault
from ..multi_fidelity.promotion_policy import SyncPromotionPolicy
from ..multi_fidelity.sampling_policy import EnsemblePolicy
from .utils import compute_config_dist


class PriorBand(HyperbandCustomDefault):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = EnsemblePolicy,
        promotion_policy: typing.Any = SyncPromotionPolicy,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None,
        ignore_errors: bool = False,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
        random_interleave_prob: float = 0.0,
        sample_default_first: bool = True,
        inc_sample_type: str = "hypersphere",  # could be "gaussian" too
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_default_first=sample_default_first,
        )
        self.inc_sample_type = inc_sample_type
        self.sampling_policy = sampling_policy(pipeline_space, self.inc_sample_type)
        self.sampling_args = {
            "inc": None,
            "weights": {
                "prior": 1,  # begin with only prior sampling
                "inc": 0,
                "random": 0,
            },
        }
        for _, sh in self.sh_brackets.items():
            sh.sampling_policy = self.sampling_policy
            sh.sampling_args = self.sampling_args
        self.rung_histories = None

    def _load_previous_observations(
        self, previous_results: dict[str, ConfigResult]
    ) -> None:
        for config_id, config_val in previous_results.items():
            _config, _rung = self._get_config_id_split(config_id)
            perf = self.get_loss(config_val.result)
            if int(_config) in self.observed_configs.index:
                # config already recorded in dataframe
                rung_recorded = self.observed_configs.at[int(_config), "rung"]
                if rung_recorded < int(_rung):
                    # config recorded for a lower rung but higher rung eval available
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = perf
            else:
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), perf]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
            # for efficiency, redefining the function to have the
            # `rung_histories` assignment inside the for loop
            self.rung_histories["config"][int(_rung)].append(int(_config))
            self.rung_histories["perf"][int(_rung)].append(perf)
        return

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        self.rung_histories = {
            "config": {rung: [] for rung in range(self.min_rung, self.max_rung + 1)},
            "perf": {rung: [] for rung in range(self.min_rung, self.max_rung + 1)},
        }
        super().load_results(previous_results, pending_evaluations)

    def find_all_distances_from_incumbent(self, incumbent):
        """Finds the distance to the nearest neighbour."""
        dist = lambda x: compute_config_dist(incumbent, x)
        # computing distance of incumbent from all seen points in history
        distances = [dist(config) for config in self.observed_configs.config]
        # ensuring the distances exclude 0 or the distance from itself
        distances = [d for d in distances if d > 0]
        return distances

    def find_1nn_distance_from_incumbent(self, incumbent):
        """Finds the distance to the nearest neighbour."""
        distances = self.find_all_distances_from_incumbent(incumbent)
        distance = min(distances)
        return distance

    def find_incumbent(self, rung: int = None) -> SearchSpace:
        """Find the best performing configuration seen so far."""
        rungs = self.observed_configs.rung.values
        idxs = self.observed_configs.index.values
        while rung is not None:
            # enters this scope is `rung` argument passed and not left empty or None
            if rung not in rungs:
                self.logger.warn(f"{rung} not in {np.unique(idxs)}")
            # filtering by rung based on argument passed
            idxs = self.observed_configs.rung.values == rung
            # checking width of current rung
            if len(idxs) < self.eta:
                self.logger.warn(
                    f"Selecting incumbent from a rung with width less than {self.eta}"
                )
        # extracting the incumbent configuration
        if len(idxs):
            # finding the config with the lowest recorded performance
            inc_idx = np.nanargmin(self.observed_configs.loc[idxs].perf.values)
            inc = self.observed_configs.loc[idxs].iloc[inc_idx].config
        else:
            # THIS block should not ever execute, but for runtime anomalies, if no
            # incumbent can be extracted, the prior is treated as the incumbent
            inc = self.pipeline_space.sample_default_configuration()
            self.logger.warn(
                "Treating the prior as the incumbent. "
                "Please check if this should not happen."
            )
        return inc

    def calc_sampling_args(self, rung_size, inc=None) -> dict:
        """Sets the sampling args for the EnsemblePolicy."""
        nincs = 0 if inc is None else self.eta
        nincs = 1 if rung_size <= self.eta else nincs
        npriors = np.floor(rung_size / self.eta)
        npriors = npriors if npriors else 1
        nrandom = rung_size - npriors - nincs
        if self.current_sh_bracket == 0 and len(self.observed_configs) < npriors:
            # Enforce only prior based samples till the required number of prior samples
            # seen at the base rung of the first ever SH bracket
            nrandom = 0
        elif (
            self.current_sh_bracket == 0
            and len(self.observed_configs) >= npriors
            and len(self.observed_configs) <= rung_size
        ):
            # Enforce only random samples when the required number of prior samples
            # at the base rung of the first ever SH bracket is seen
            nrandom = 1
            npriors = 0
        if self.current_sh_bracket == len(self.sh_brackets) - 1:
            # disable random search at the max rung
            nrandom = 0
            npriors = self.eta * nincs
        # normalize weights into probabilities
        _total = npriors + nincs + nrandom
        sampling_args = {
            "inc": inc,
            "weights": {
                "prior": npriors / _total,
                "inc": nincs / _total,
                "random": nrandom / _total,
            },
        }
        return sampling_args

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        sh_bracket = self.sh_brackets[self.current_sh_bracket]
        rung_size = sh_bracket.config_map[sh_bracket.min_rung]
        # don't activate incumbent for the base rung in the first bracket
        if self.current_sh_bracket == 0 and len(self.observed_configs) <= rung_size:
            # no incumbent selection yet
            inc = None
        else:
            inc = self.find_incumbent()
        self.sampling_args = self.calc_sampling_args(rung_size, inc)
        if self.inc_sample_type == "hypersphere" and inc is not None:
            min_dist = self.find_all_distances_from_incumbent(
                inc
            )  # self.find_1nn_distance_from_incumbent(inc)
            self.sampling_args.update({"distance": min_dist})
        for _, sh in self.sh_brackets.items():
            sh.sampling_args = self.sampling_args
        return super().get_config_and_ids()


class PriorBandCustom(PriorBand):
    def calc_sampling_args(self, rung_size, inc=None) -> dict:
        sampling_args = {
            "inc": inc,
            "weights": {
                "prior": 1 / 2 if inc is None else 1 / 3,
                "inc": 0 if inc is None else 1 / 3,
                "random": 1 / 2 if inc is None else 1 / 3,
            },
        }
        return sampling_args
