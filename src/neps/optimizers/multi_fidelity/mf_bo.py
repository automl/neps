from __future__ import annotations


class MFBOBase:
    def _active_rung(self):
        rung = self.max_rung
        while rung >= 0:
            if len(self.rung_histories[rung]["config"]) > self.init_size:
                return rung
            rung -= 1
        return None

    def _fit_models(self):
        # define any model or surrogate training and acquisition function state setting
        # if adding model-based search to the basic multi-fidelity algorithm

        if not self.is_init_phase():
            # inside here rung should not be None
            rung = self._active_rung()
            train_df = self.observed_configs.loc[self.observed_configs.rung == rung]
            self.model_policy.update_model(train_df.config, train_df.perf)

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
