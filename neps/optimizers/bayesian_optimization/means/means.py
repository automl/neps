from __future__ import annotations

import gpytorch
import torch
from metahyper.utils import instance_from_map


class HpMean:
    def __init__(self, active_hps: list[str] = None):
        self.active_hps = active_hps
        self.active_dims = None

    def build(self, hp_shapes):
        active_dims = []
        for hp_name in self.active_hps:
            active_dims.extend(hp_shapes[hp_name].active_dims)
        self.active_dims = tuple(active_dims)


class ZeroMean(HpMean):
    def build(self, hp_shapes):
        super().build(hp_shapes)
        return gpytorch.means.ZeroMean()


class ConstantMean(HpMean):
    def build(self, hp_shapes):
        super().build(hp_shapes)
        return gpytorch.means.ConstantMean()


class LinearMean(HpMean):
    def build(self, hp_shapes):
        super().build(hp_shapes)
        return gpytorch.means.LinearMean(len(self.active_dims))


class GptMeanComposer(gpytorch.means.Mean):
    def __init__(self, means_locations):
        super().__init__()
        self.means_locations = means_locations
        # Store the means as attributes to be included in pytorch variables
        for i, (_, mean) in enumerate(means_locations):
            setattr(self, f"mean_{i}", mean)

    def forward(self, x):
        resulting_mean = torch.zeros(x.shape[0])
        for idx, mean in self.means_locations:
            columns = [x[:, i] for i in idx]
            columns = torch.stack(columns, axis=1)
            resulting_mean += mean(columns)
        return resulting_mean


class MeanComposer:
    def __init__(self, pipeline_space, *means, fallback_default_mean="constant"):
        available_hps = set(pipeline_space.keys())
        self.means = [instance_from_map(MEANS_MAPPING, m, "mean") for m in means]

        # Ensure every hp is used only one time
        default_mean = None
        for m in self.means:
            if m.active_hps is not None:
                for hp_name in m.active_hps:
                    if hp_name in available_hps:
                        available_hps.discard(hp_name)
                    elif hp_name in pipeline_space:
                        raise Exception(
                            f"Can't use the same hyperparameter ({hp_name}) with multiple means"
                        )
                    else:
                        raise ValueError(f"No hyperparameter named {hp_name}")
            elif default_mean is not None:
                raise Exception(
                    f"Can't have multiple default mean functions ({default_mean} and {m})"
                )
            else:
                default_mean = m

        # Assign default mean
        if default_mean is None:
            default_mean = instance_from_map(MEANS_MAPPING, fallback_default_mean, "mean")
            self.means.append(default_mean)
        default_mean.active_hps = list(available_hps)

    def build(self, hp_shapes):
        sub_means = []
        for m in self.means:
            gpt_mean = m.build(hp_shapes)
            sub_means.append((m.active_dims, gpt_mean))
        return GptMeanComposer(sub_means)


MEANS_MAPPING = {
    "zero": ZeroMean,
    "constant": ConstantMean,
    "linear": LinearMean,
}
