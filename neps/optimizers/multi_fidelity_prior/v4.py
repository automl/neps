import typing
from typing import Dict

import numpy as np
import pandas as pd
from metahyper.api import ConfigResult
from scipy.stats import uniform
from typing_extensions import Literal

from ...search_spaces.hyperparameters.categorical import CategoricalParameter
from ...search_spaces.hyperparameters.float import FloatParameter
from ...search_spaces.hyperparameters.integer import IntegerParameter
from ...search_spaces.search_space import SearchSpace
from ..multi_fidelity.promotion_policy import AsyncPromotionPolicy
from ..multi_fidelity.sampling_policy import FixedPriorPolicy
from ..multi_fidelity.successive_halving import SuccessiveHalvingWithPriors


def _compute_uniform(low, high, log: bool = False):
    u = uniform(loc=low, scale=high)
    return np.log(u.pdf(u.rvs()) + 1e-12) if log else u.pdf(u.rvs())


class OurOptimizerV4(SuccessiveHalvingWithPriors):
    """Implements SH with priors where priors decay to uniform per rung level."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: typing.Any = FixedPriorPolicy,
        promotion_policy: typing.Any = AsyncPromotionPolicy,
        loss_value_on_error: float = None,
        cost_value_on_error: float = None,
        logger=None,
        prior_confidence: Literal["low", "medium", "high"] = "medium",
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            early_stopping_rate=early_stopping_rate,
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
            prior_confidence=prior_confidence,
        )
        # keeps a count of number of function evaluations made at each rung level
        self.rung_visits: Dict[int, int] = dict()

    def _load_previous_observations(
        self, previous_results: Dict[str, ConfigResult]
    ) -> None:
        # duplicating code from SuccessiveHalving to not parse previous_results again
        self.rung_visits = {rung: 0 for rung in self.rung_map.keys()}
        for config_id, config_val in previous_results.items():
            _config, _rung = self._get_config_id_split(config_id)
            # only change from SuccessiveHalving
            self.rung_visits[int(_rung)] += 1
            if int(_config) in self.observed_configs.index:
                # config already recorded in dataframe
                rung_recorded = self.observed_configs.at[int(_config), "rung"]
                if rung_recorded < int(_rung):
                    # config recorded for a lower rung but higher rung eval available
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = self.get_loss(
                        config_val.result
                    )
            else:
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), self.get_loss(config_val.result)]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
        return

    def sample_new_config(self, **kwargs):
        npriors = 1
        nrandom = 10
        configs = [
            self.pipeline_space.sample(
                patience=self.patience,
                user_priors=False,
                ignore_fidelity=True,
            )
            for _ in range(nrandom)
        ]
        configs.extend(
            [
                self.pipeline_space.sample(
                    patience=self.patience,
                    user_priors=True,
                    ignore_fidelity=True,
                )
                for _ in range(npriors)
            ]
        )

        rung = kwargs["rung"] if "rung" in kwargs else 0
        alpha = 0.95
        # polyak averaging --- slowly going towards uniform distribution
        polyak = lambda p_x, u_x, alpha: alpha * p_x + (1 - alpha) * u_x
        scores = []
        for config in configs:
            p_x = config.compute_prior()
            u_x = self.compute_uniform_prior(config)
            for _ in range(self.rung_visits[rung] - 1):
                p_x = polyak(p_x, u_x, alpha)
            scores.append(p_x)
        idx = np.argmax(scores)
        config = configs[idx]
        return config

    def compute_uniform_prior(self, config, log: bool = False):
        density_value = 0.0 if log else 1.0
        for hyperparameter in config.values():
            if isinstance(hyperparameter, (FloatParameter, IntegerParameter)):
                low, high, _ = hyperparameter._get_low_high_default()
            elif isinstance(hyperparameter, CategoricalParameter):
                low = 0
                high = len(hyperparameter.choices)
            if hasattr(hyperparameter, "has_prior") and hyperparameter.has_prior:
                if log:
                    density_value += _compute_uniform(low, high, log=True)
                else:
                    density_value *= _compute_uniform(low, high, log=False)
        return density_value
