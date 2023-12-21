# type: ignore
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from ...metahyper import instance_from_map
from ..bayesian_optimization.models import SurrogateModelMapping
from ..multi_fidelity.utils import normalize_vectorize_config
from ..multi_fidelity_prior.utils import calc_total_resources_spent, update_fidelity
from ..utils import map_real_hyperparameters_from_tabular_ids


class MFBOBase:
    """Designed to work with model-based search on SH-based multi-fidelity algorithms.

    Requires certain strict assumptions about fidelities and rung maps.
    """

    def _fit_models(self):
        """Performs necessary procedures to build and use models."""

        if not self.model_based:
            # do nothing here if the algorithm has model-based search disabled
            return

        if self.is_init_phase():
            return

        if self.pipeline_space.has_prior:
            # PriorBand + BO
            total_resources = calc_total_resources_spent(
                self.observed_configs, self.rung_map
            )
            decay_t = total_resources / self.max_budget
        else:
            # Mobster
            decay_t = None

        # extract pending configurations
        # doing this separately as `rung_histories` do not record pending configs
        pending_df = self.observed_configs[self.observed_configs.perf.isna()]
        if self.modelling_type == "rung":
            # collect only the finished configurations at the highest active `rung`
            # for training the surrogate and considering only those pending
            # evaluations at `rung` for fantasization
            # important to set the fidelity value of the training data configuration
            # such that the fidelity variable is rendered ineffective in the model

            rung = self._active_rung()
            # inside here rung should not be None
            if rung is None:
                raise ValueError(
                    "Returned rung is None. Should not be so when not init phase."
                )
            self.logger.info(f"Building model at rung {rung}")
            # collecting finished evaluations at `rung`
            train_df = self.observed_configs.loc[
                self.rung_histories[rung]["config"]
            ].copy()

            # setting the fidelity value and performance to match the rung history
            # a promoted configuration may have a different fidelity than the
            # rung history recorded
            fidelities = [self.rung_map[rung]] * len(train_df)
            train_x = deepcopy(train_df.config.values.tolist())
            # update fidelity
            train_x = list(map(update_fidelity, train_x, fidelities))
            train_y = deepcopy(self.rung_histories[rung]["perf"])
            # extract only the pending configurations that are at `rung`
            pending_df = pending_df[pending_df.rung == rung]
            pending_x = deepcopy(pending_df.config.values.tolist())
            # update fidelity
            fidelities = [self.rung_map[rung]] * len(pending_x)
            pending_x = list(map(update_fidelity, pending_x, fidelities))

        elif self.modelling_type == "joint":
            # collect ALL configurations ever recorded for training the surrogate
            # and considering all pending evaluations for fantasization
            # the fidelity for all these configurations should be set for each of the
            # rungs they were evaluated at in the entire optimization history

            # NOTE: pandas considers mutable objects inside dataframes as antipattern
            train_x = []
            train_y = []
            pending_x = []
            for rung in range(self.min_rung, self.max_rung + 1):
                _ids = self.rung_histories[rung]["config"]
                _x = deepcopy(self.observed_configs.loc[_ids].config.values.tolist())
                # update fidelity
                fidelity = [self.rung_map[rung]] * len(_x)
                _x = list(map(update_fidelity, _x, fidelity))
                _y = deepcopy(self.rung_histories[rung]["perf"])
                train_x.extend(_x)
                train_y.extend(_y)
            # setting the fidelity value of the pending configs appropriately
            get_fidelity = lambda _rung: self.rung_map[_rung]
            fidelities = list(map(get_fidelity, pending_df.rung.values))
            _pending_x = list(
                map(update_fidelity, pending_df.config.values.tolist(), fidelities)
            )
            pending_x.extend(_pending_x)
        else:
            raise ValueError("Choice of modelling_type not in {{'rung', 'joint'}}")
        # the `model_policy` class should define a function to train the surrogates
        # and set the acquisition states
        self.model_policy.update_model(train_x, train_y, pending_x, decay_t=decay_t)

    def _active_rung(self):
        """Returns the highest rung that can fit a model, `None` if no rung is eligible."""
        rung = self.max_rung
        while rung >= self.min_rung:
            if len(self.rung_histories[rung]["config"]) >= self.init_size:
                return rung
            rung -= 1
        return None

    def is_init_phase(self) -> bool:
        """Returns True is in the warmstart phase and False under model-based search."""
        if self.modelling_type == "rung":
            # build a model per rung or per fidelity
            # in this case, the initial design checks if `init_size` number of
            # configurations have finished at a rung or not and the highest such rung is
            # chosen for model building at teh current iteration
            if self._active_rung() is None:
                return True
        elif self.modelling_type == "joint":
            # builds a model across all fidelities with the fidelity as a dimension
            # in this case, calculate the total number of function evaluations spent
            # and in vanilla BO fashion use that to compare with the initital design size
            resources = calc_total_resources_spent(self.observed_configs, self.rung_map)
            resources /= self.max_budget
            if resources < self.init_size:
                return True
        else:
            raise ValueError("Choice of modelling_type not in {{'rung', 'joint'}}")
        return False

    def sample_new_config(
        self,
        rung: int = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Samples configuration from policies or random."""
        if self.model_based and not self.is_init_phase():
            incumbent = None
            if self.modelling_type == "rung":
                # `rung` should not be None when not in init phase
                active_max_rung = self._active_rung()
                fidelity = None
                active_max_fidelity = self.rung_map[active_max_rung]
            elif self.modelling_type == "joint":
                fidelity = self.rung_map[rung]
                active_max_fidelity = None
                # IMPORTANT step for correct 2-step acquisition
                incumbent = min(self.rung_histories[rung]["perf"])
            else:
                fidelity = active_max_fidelity = None
            assert (
                (fidelity is None and active_max_fidelity is not None)
                or (active_max_fidelity is None and fidelity is not None)
                or (active_max_fidelity is not None and fidelity is not None)
            ), "Either condition needs to be not None!"
            config = self.model_policy.sample(
                active_max_fidelity=active_max_fidelity,
                fidelity=fidelity,
                incumbent=incumbent,
                **self.sampling_args,
            )
        elif self.sampling_policy is not None:
            config = self.sampling_policy.sample(**self.sampling_args)
        else:
            config = self.pipeline_space.sample(
                patience=self.patience,
                user_priors=self.use_priors,
                ignore_fidelity=True,
            )
        return config


class FreezeThawModel:
    """Designed to work with model search in unit step multi-fidelity algorithms."""

    def __init__(
        self,
        pipeline_space,
        surrogate_model: str = "deep_gp",
        surrogate_model_args: dict = None,
    ):
        self.observed_configs = None
        self.pipeline_space = pipeline_space
        self.surrogate_model_name = surrogate_model
        self.surrogate_model_args = (
            surrogate_model_args if surrogate_model_args is not None else {}
        )
        if self.surrogate_model_name in ["deep_gp", "pfn"]:
            self.surrogate_model_args.update({"pipeline_space": pipeline_space})

        # instantiate the surrogate model
        self.surrogate_model = instance_from_map(
            SurrogateModelMapping,
            self.surrogate_model_name,
            name="surrogate model",
            kwargs=self.surrogate_model_args,
        )

    def _fantasize_pending(self, train_x, train_y, pending_x):
        # Select configs that are neither pending nor resulted in error
        completed_configs = self.observed_configs.completed_runs.copy(deep=True)
        # IMPORTANT: preprocess observations to get appropriate training data
        train_x, train_lcs, train_y = self.observed_configs.get_training_data_4DyHPO(
            completed_configs, self.pipeline_space
        )
        pending_condition = self.observed_configs.pending_condition
        if pending_condition.any():
            pending_configs = self.observed_configs.df.loc[pending_condition]
            pending_x, pending_lcs, _ = self.observed_configs.get_training_data_4DyHPO(
                pending_configs
            )
            self._fit(train_x, train_y, train_lcs)
            _y, _ = self._predict(pending_x, pending_lcs)
            _y = _y.tolist()

            train_x.extend(pending_x)
            train_y.extend(_y)
            train_lcs.extend(pending_lcs)

        return train_x, train_y, train_lcs

    def _fit(self, train_x, train_y, train_lcs):
        if self.surrogate_model_name in ["gp", "gp_hierarchy"]:
            self.surrogate_model.fit(train_x, train_y)
        elif self.surrogate_model_name in ["deep_gp", "pfn"]:
            self.surrogate_model.fit(train_x, train_y, train_lcs)
        else:
            # check neps/optimizers/bayesian_optimization/models/__init__.py for options
            raise ValueError(
                f"Surrogate model {self.surrogate_model_name} not supported!"
            )

    def _predict(self, test_x, test_lcs):
        if self.surrogate_model_name in ["gp", "gp_hierarchy"]:
            return self.surrogate_model.predict(test_x)
        elif self.surrogate_model_name in ["deep_gp", "pfn"]:
            return self.surrogate_model.predict(test_x, test_lcs)
        else:
            # check neps/optimizers/bayesian_optimization/models/__init__.py for options
            raise ValueError(
                f"Surrogate model {self.surrogate_model_name} not supported!"
            )

    def set_state(
        self,
        pipeline_space,
        surrogate_model_args,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.pipeline_space = pipeline_space
        self.surrogate_model_args = (
            surrogate_model_args if surrogate_model_args is not None else {}
        )
        # only to handle tabular spaces
        if self.pipeline_space.has_tabular:
            if self.surrogate_model_name in ["deep_gp", "pfn"]:
                self.surrogate_model_args.update(
                    {"pipeline_space": self.pipeline_space.raw_tabular_space}
                )
            # instantiate the surrogate model, again, with the new pipeline space
            self.surrogate_model = instance_from_map(
                SurrogateModelMapping,
                self.surrogate_model_name,
                name="surrogate model",
                kwargs=self.surrogate_model_args,
            )

    def update_model(self, train_x=None, train_y=None, pending_x=None, decay_t=None):
        if train_x is None:
            train_x = []
        if train_y is None:
            train_y = []
        if pending_x is None:
            pending_x = []

        if decay_t is None:
            decay_t = len(train_x)
        train_x, train_y, train_lcs = self._fantasize_pending(train_x, train_y, pending_x)
        self._fit(train_x, train_y, train_lcs)

        return self.surrogate_model, decay_t


class PFNSurrogate(FreezeThawModel):
    """Special class to deal with PFN surrogate model and freeze-thaw acquisition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_x = None
        self.train_y = None

    def _fit(self, *args):  # pylint: disable=unused-argument
        assert self.surrogate_model_name == "pfn"
        self.preprocess_training_set()
        self.surrogate_model.fit(self.train_x, self.train_y)

    def preprocess_training_set(self):
        _configs = self.observed_configs.df.config.values.copy()

        # onlf if tabular space is present
        if self.pipeline_space.has_tabular:
            # placeholder index, will be driooed
            _idxs = np.arange(len(_configs))
            # mapping the (id, epoch) space of tabular configs to the actual HPs
            _configs = map_real_hyperparameters_from_tabular_ids(
                pd.Series(_configs, index=_idxs), self.pipeline_space
            ).values

        device = self.surrogate_model.device
        # TODO: fix or make consistent with `tokenize``
        configs, idxs, performances = self.observed_configs.get_tokenized_data(
            self.observed_configs.df.copy().assign(config=_configs)
        )
        # TODO: account for fantasization
        self.train_x = torch.Tensor(np.hstack([idxs, configs])).to(device)
        self.train_y = torch.Tensor(performances).to(device)

    def preprocess_test_set(self, test_x):
        _len = len(self.observed_configs.all_configs_list())
        device = self.surrogate_model.device

        new_idxs = np.arange(_len, len(test_x))
        base_fidelity = np.array([1] * len(new_idxs))
        new_token_ids = np.hstack(
            (new_idxs.T.reshape(-1, 1), base_fidelity.T.reshape(-1, 1))
        )
        # the following operation takes each element in the array and stacks it vertically
        # in this case, should convert a (n,) array to (n, 2) by flattening the elements
        existing_token_ids = np.vstack(self.observed_configs.token_ids).astype(int)
        token_ids = np.vstack((existing_token_ids, new_token_ids))

        configs = np.array([normalize_vectorize_config(c) for c in test_x])
        test_x = torch.Tensor(np.hstack([token_ids, configs])).to(device)
        return test_x

    def _predict(self, test_x, test_lcs):
        assert self.surrogate_model_name == "pfn"
        test_x = self.preprocess_test_set(test_x)
        return self.surrogate_model.predict(self.train_x, self.train_y, test_x)
