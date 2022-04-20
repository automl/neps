from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pandas as pd
from metahyper.api import ConfigResult, instance_from_map

from ...search_spaces import (
    CategoricalParameter,
    FloatParameter,
    GraphGrammar,
    IntegerParameter,
)
from ...search_spaces.search_space import SearchSpace
from ...utils.common import has_instance
from ...utils.result_utils import get_loss
from ..base_optimizer import BaseOptimizer
from .acquisition_functions import AcquisitionMapping
from .acquisition_functions.prior_weighted import DecayingPriorWeightedAcquisition
from .acquisition_samplers import AcquisitionSamplerMapping
from .kernels import GraphKernelMapping, StationaryKernelMapping
from .models import SurrogateModelMapping


class BayesianOptimization(BaseOptimizer):
    """Implements the basic BO loop."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        initial_design_size: int = 10,
        surrogate_model: str | Any = "gp",
        surrogate_model_fit_args: dict = None,
        optimal_assignment: bool = False,
        domain_se_kernel: str = None,
        graph_kernels: list = None,
        hp_kernels: list = None,
        acquisition: str | Any = "EI",
        log_prior_weighted: bool = False,
        acquisition_sampler: str | Any = "mutation",
        random_interleave_prob: float = 0.0,
        patience: int = 100,
        budget: None | int | float = None,
        logger=None,
    ):
        """Initialise the BO loop.

        Args:
            pipeline_space: Space in which to search
            initial_design_size: Number of 'x' samples that need to be evaluated before
                selecting a sample using a strategy instead of randomly.
            surrogate_model: Surrogate model
            surrogate_model_fit_args: Arguments that will be given to the surrogate model
                (the Gaussian processes model).
            optimal_assignment: whether the optimal assignment kernel should be used.
            domain_se_kernel: Stationary kernel name
            graph_kernels: Kernels for NAS
            hp_kernels: Kernels for HPO
            acquisition: Acquisition strategy
            log_prior_weighted: if to use log for prior
            acquisition_sampler: Acquisition function fetching strategy
            random_interleave_prob: Frequency at which random configurations are sampled
                instead of configurations from the acquisition strategy.
            patience: How many times we try something that fails before giving up.
            budget: Maximum budget
            logger: logger object, or None to use the neps logger

        Raises:
            ValueError: if patience < 1
            ValueError: if initial_design_size < 1
            ValueError: if random_interleave_prob is not between 0.0 and 1.0
            ValueError: if no kernel is provided
        """
        super().__init__(
            pipeline_space=pipeline_space,
            patience=patience,
            logger=logger,
            budget=budget,
        )

        if initial_design_size < 1:
            raise ValueError(
                "BayesianOptimization needs initial_design_size to be at least 1"
            )
        if not 0 <= random_interleave_prob <= 1:
            raise ValueError("random_interleave_prob should be between 0.0 and 1.0")

        self._initial_design_size = initial_design_size
        self._random_interleave_prob = random_interleave_prob
        self._num_train_x: int
        self._pending_evaluations: list = []
        self._model_update_failed = False

        if not graph_kernels:
            graph_kernels = []
            if has_instance(self.pipeline_space.values(), GraphGrammar):
                graph_kernels.append("wl")

        if not hp_kernels:
            hp_kernels = []
            if has_instance(
                self.pipeline_space.values(), FloatParameter, IntegerParameter
            ):
                hp_kernels.append("m52")
            if has_instance(self.pipeline_space.values(), CategoricalParameter):
                hp_kernels.append("hm")

        graph_kernels = [
            instance_from_map(GraphKernelMapping, kernel, "kernel", as_class=True)(
                oa=optimal_assignment,
                se_kernel=instance_from_map(
                    StationaryKernelMapping, domain_se_kernel, "se kernel"
                ),
            )
            for kernel in graph_kernels
        ]
        hp_kernels = [
            instance_from_map(StationaryKernelMapping, kernel, "kernel")
            for kernel in hp_kernels
        ]

        if not graph_kernels and not hp_kernels:
            raise ValueError("No kernels are provided!")

        self.surrogate_model = instance_from_map(
            SurrogateModelMapping,
            surrogate_model,
            name="surrogate model",
            kwargs={
                "graph_kernels": graph_kernels,
                "hp_kernels": hp_kernels,
                "vectorial_features": self.pipeline_space.get_vectorial_dim(),
                "surrogate_model_fit_args": surrogate_model_fit_args or {},
            },
        )
        self.acquisition = instance_from_map(
            AcquisitionMapping,
            acquisition,
            name="acquisition function",
        )
        # TODO: Do we want to apply this everytime?
        self.acquisition = DecayingPriorWeightedAcquisition(
            self.acquisition, log=log_prior_weighted
        )
        self.acquisition_sampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,
            name="acquisition sampler function",
            kwargs={"patience": self.patience, "pipeline_space": self.pipeline_space},
        )

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        # TODO: filter out error configs as they can not be used for modeling?
        # TODO: read out cost if they exist
        train_x = [el.config for el in previous_results.values()]
        train_y = [get_loss(el.result) for el in previous_results.values()]
        self._num_train_x = len(train_x)
        self._pending_evaluations = [el for el in pending_evaluations.values()]
        if self._num_train_x >= self._initial_design_size:
            try:
                if len(self._pending_evaluations) > 0:
                    # We want to use hallucinated results for the evaluations that have
                    # not finished yet. For this we fit a model on the finished
                    # evaluations and add these to the other results to fit another model.
                    self.surrogate_model.fit(train_x, train_y)
                    ys, _ = self.surrogate_model.predict(self._pending_evaluations)
                    train_x += self._pending_evaluations
                    train_y += list(ys.detach().numpy())

                self.surrogate_model.fit(train_x, train_y)
                self.acquisition.set_state(self.surrogate_model)
                self.acquisition_sampler.set_state(x=train_x, y=train_y)

                self._model_update_failed = False
            except RuntimeError:
                self.logger.exception(
                    "Model could not be updated due to below error. Sampling will not use"
                    " the model."
                )
                self._model_update_failed = True

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        if self._num_train_x == 0 and self._initial_design_size >= 1:
            # TODO: if default config sample it
            config = self.pipeline_space.sample(patience=self.patience, user_priors=True)
        elif random.random() < self._random_interleave_prob:
            config = self.pipeline_space.sample(patience=self.patience)
        elif self._num_train_x < self._initial_design_size or self._model_update_failed:
            config = self.pipeline_space.sample(patience=self.patience, user_priors=True)
        else:
            for _ in range(self.patience):
                config = self.acquisition_sampler.sample(self.acquisition)
                if config not in self._pending_evaluations:
                    break
            else:
                config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True
                )

        config_id = str(self._num_train_x + len(self._pending_evaluations) + 1)
        return config, config_id, None


# TODO: Update according to the changes above
# TODO(neps.api): this BO class gets used when
# pipeline_space.has_fidelity() == True and BO is chosen
# also when random_search is chosen, but then use no model
class BayesianOptimizationMultiFidelity(BayesianOptimization):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        initial_design_size: int = 10,
        surrogate_model_fit_args: dict = None,
        optimal_assignment: bool = False,
        domain_se_kernel: str = None,
        graph_kernels: list = None,
        hp_kernels: list = None,
        acquisition: str = "EI",
        random_interleave_prob: float = 0.0,
        patience: int = 50,
        eta: int = 4,
        early_stopping_rate: int = 0,
        initial_design_type: str = "unique_configs",
        logger=None,
        **kwargs,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            initial_design_size=initial_design_size,
            surrogate_model_fit_args=surrogate_model_fit_args,
            optimal_assignment=optimal_assignment,
            domain_se_kernel=domain_se_kernel,
            graph_kernels=graph_kernels,
            hp_kernels=hp_kernels,
            acquisition=acquisition,
            random_interleave_prob=random_interleave_prob,
            patience=patience,
            logger=logger,
            **kwargs,
        )
        self.min_budget = pipeline_space.fidelity.lower
        self.max_budget = pipeline_space.fidelity.upper
        self.eta = eta
        self.early_stopping_rate = early_stopping_rate
        # `max_budget_init` checks for the number of configurations that have been
        # evaluated at the target budget
        self.initial_design_type = initial_design_type

        # check to ensure no rung ID is negative
        self.stopping_rate_limit = np.floor(
            np.log(self.max_budget / self.min_budget) / np.log(self.eta)
        ).astype(int)
        assert self.early_stopping_rate <= self.stopping_rate_limit

        # maps rungs to a fidelity value
        self.rung_map = self._get_rung_map(self.early_stopping_rate)
        self.max_rung = len(self.rung_map) - 1
        self.fidelities = list(self.rung_map.values())
        # stores the observations made and the corresponding fidelity explored
        # crucial data structure used for determining promotion candidates
        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))
        # stores which configs occupy each rung at any time
        self.rung_members = {k: [] for k in self.rung_map.keys()}  # type: Dict
        # one-to-one with self.rung_members, storing corresponding performance
        self.rung_members_performance = {
            k: [] for k in self.rung_map.keys()
        }  # type: Dict
        self.rung_promotions = dict()  # type: Dict
        self.total_fevals = 0  # type: int

    def _get_rung_map(self, s: int = 0) -> dict:
        """Maps rungs (0,1,...,k) to a fidelity value based on fidelity bounds, eta, s."""
        assert s <= self.stopping_rate_limit
        new_min_budget = self.min_budget * (self.eta**s)
        nrungs = (
            np.floor(np.log(self.max_budget / new_min_budget) / np.log(self.eta)).astype(
                int
            )
            + 1
        )
        rung_map = dict()
        for i in range(nrungs):
            rung_map[i] = (
                int(new_min_budget)
                if isinstance(self.pipeline_space.fidelity, IntegerParameter)
                else new_min_budget
            )
            new_min_budget *= self.eta
        return rung_map

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        # TODO: Read in rungs using the config id (alternatively, use get/load state)
        super().load_results(previous_results, pending_evaluations)

        self.total_fevals = len(previous_results)
        # iterates over all previous results and updates the list of observed
        # configs with the highest fidelity it was evaluated on and its performance
        for config_id, _ in previous_results.items():
            # any config occurring in `previous_results` should be in `observed_configs`
            _config, _rung = config_id.split("_")
            # `max` is important to keep track of the performance of a configuration on
            # the highest fidelity seen as `previous_results` contain all evaluations
            _rung = max(int(_rung), self.observed_configs.iloc[int(_config)]["rung"])
            # _config = int(_config)
            self.observed_configs.at[int(_config), "rung"] = _rung
            perf = get_loss(previous_results[f"{int(_config)}_{_rung}"].result)
            self.observed_configs.at[int(_config), "perf"] = perf
        # iterates over the list of explored configs and buckets them to respective
        # rungs depending on the highest fidelity it was evaluated at
        self.rung_members = {k: [] for k in range(self.max_rung)}
        self.rung_members_performance = {k: [] for k in range(self.max_rung)}
        for _rung in self.observed_configs.rung.unique():
            self.rung_members[_rung] = self.observed_configs.index[
                self.observed_configs.rung == _rung
            ].values
            self.rung_members_performance[_rung] = self.observed_configs.perf[
                self.observed_configs.rung == _rung
            ].values
        # identifying promotion list per rung
        self.rung_promotions = dict()
        for _rung in self.rung_map.keys():
            if _rung == self.max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            top_k = len(self.rung_members_performance[_rung]) // self.eta
            self.rung_promotions[_rung] = np.array(self.rung_members[_rung])[
                np.argsort(self.rung_members_performance[_rung])[:top_k]
            ].tolist()
        return

    def is_promotable(self) -> int | None:
        rung_to_promote = None
        for _rung in reversed(range(self.max_rung)):
            if len(self.rung_promotions[_rung]) > 0:
                rung_to_promote = _rung
                break
        return rung_to_promote

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase/model-based search.

        If `initial_design_type` is `max_budget`, the number of evaluations made at the
            target budget is compared to `initial_design_size`.
        If `initial_design_type` is `unique_configs`, the total number of unique
            configurations is compared to the `initial_design_size`.
        """
        if self.initial_design_type == "max_budget":
            val = (
                np.sum(self.observed_configs.rung == self.max_rung)
                < self._initial_design_size
            )
        else:
            val = len(self.observed_configs) <= self._initial_design_size
        return val

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        rung_to_promote = self.is_promotable()
        if rung_to_promote is not None:
            # promote existing config
            row = self.observed_configs.iloc[self.rung_promotions[rung_to_promote][0]]
            config = deepcopy(row["config"])
            rung = rung_to_promote + 1
            # assigning the fidelity to evaluate the config at
            config.fidelity.value = self.rung_map[rung]
            # updating config IDs
            previous_config_id = f"{row.name}_{rung_to_promote}"
            config_id = f"{row.name}_{rung}"
            # updating observation tracker
            self.observed_configs.at[row.name, "rung"] = rung
        else:
            if self.is_init_phase():
                # random sampling a config at base rung
                config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True
                )
            else:
                # sampling from AF at base rung
                config = self.acquisition_sampler.sample(self.acquisition)
            # assigning the fidelity to evaluate the config at
            config.fidelity.value = self.rung_map[0]  # base rung is always 0
            # updating observation tracker
            _df = pd.DataFrame(
                [[config, 0, None]],
                columns=self.observed_configs.columns,
                index=pd.Series(len(self.observed_configs)),  # key for config_id
            )
            self.observed_configs = pd.concat((self.observed_configs, _df))
            # updating config IDs
            config_id = f"{len(self.observed_configs) - 1}_{0}"
            previous_config_id = None
        return config, config_id, previous_config_id  # type: ignore
