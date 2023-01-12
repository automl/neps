from __future__ import annotations


class MFBOBase:
    def _fit_models(self):
        # define any model or surrogate training and acquisition function state setting
        # if adding model-based search to the basic multi-fidelity algorithm

        # pylint: disable=attribute-defined-outside-init
        self.sampling_args = {
            "train_x": self.observed_configs.config.to_list(),
            "train_y": self.observed_configs.perf.to_list(),
        }
        for _, sh in self.sh_brackets.items():
            sh.sampling_args = self.sampling_args
