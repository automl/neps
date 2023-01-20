from __future__ import annotations

from ..multi_fidelity_prior.utils import calc_total_resources_spent


class MFBOBase:
    def _active_rung(self):
        rung = self.max_rung
        while rung >= self.min_rung:
            if len(self.rung_histories[rung]["config"]) > self.init_size:
                return rung
            rung -= 1
        return None

    def _fit_models(self):
        # define any model or surrogate training and acquisition function state setting
        # if adding model-based search to the basic multi-fidelity algorithm

        # TODO: what to do with decay_t
        if self.pipeline_space.has_prior:
            # PriorBand
            total_resources = calc_total_resources_spent(
                self.observed_configs, self.rung_map
            )
            decay_t = total_resources / self.max_budget
        else:
            # Mobster
            decay_t = None

        if not self.is_init_phase():
            # inside here rung should not be None
            rung = self._active_rung()
            if rung is None:
                raise ValueError(
                    "Returned rung is None. Should not be so when not init phase."
                )
            # collecting finished evaluations at `rung`
            train_df = self.observed_configs.loc[self.rung_histories[rung]["config"]]
            self.model_policy.update_model(train_df.config, train_df.perf, decay_t)

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase / model-based
        search."""
        if self._active_rung() is None:
            return True
        return False

    def sample_new_config(
        self,
        **kwargs,  # pylint: disable=unused-argument
    ):
        # Samples configuration from policy or random
        if self.model_based and not self.is_init_phase():
            config = self.model_policy.sample(**self.sampling_args)
        elif self.sampling_policy is not None:
            config = self.sampling_policy.sample(**self.sampling_args)
        else:
            config = self.pipeline_space.sample(
                patience=self.patience,
                user_priors=self.use_priors,
                ignore_fidelity=True,
            )
        return config
