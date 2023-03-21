from __future__ import annotations

from copy import deepcopy

from ..multi_fidelity_prior.utils import calc_total_resources_spent, update_fidelity


class MFBOBase:
    def _fit_models(self):
        """Performs necessary procedures to build and use models."""

        if not self.model_based:
            # do nothing here if the algorithm has model-based search disabled
            return

        if self.is_init_phase():
            return

        if self.pipeline_space.has_prior:
            # PriorBand
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
