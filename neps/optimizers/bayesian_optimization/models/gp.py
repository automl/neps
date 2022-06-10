from __future__ import annotations

from typing import Iterable

import botorch
import gpytorch
import torch
from typing_extensions import Literal

from ....search_spaces.search_space import SearchSpace
from ..default_consts import (
    DEFAULT_COMBINE,
    DEFAULT_MEAN,
    EPSILON,
    MIN_INFERRED_NOISE_LEVEL,
)
from ..kernels import Kernel, instantiate_kernel
from ..means import GpMean, MeanComposer


class GPTorchModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        mean,
        kernel,
        noise: Literal["no", "low", "high"] | float = "low",
    ):
        if noise == "no":
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
            )
            likelihood.noise = 1e-4
            likelihood.noise_covar.raw_noise.requires_grad_(False)
        else:
            if noise == "low":
                noise_level = 1e-2
            elif noise == "high":
                noise_level = 1e-1
            elif isinstance(noise, float):
                noise_level = noise
            noise_prior = gpytorch.priors.GammaPrior(1 + noise_level, noise_level / 2)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=gpytorch.constraints.GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    initial_value=noise_prior_mode,
                ),
            )

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel:
    def __init__(
        self,
        pipeline_space,
        means: Iterable[GpMean] = None,
        kernels: Iterable[Kernel] = None,
        combine_kernel: str = DEFAULT_COMBINE,
        logger=None,
        noise="low",
    ):
        self.logger = logger
        self.gp = None
        self.fitted_on = None
        self.tensor_size = None
        self.all_hp_shapes = None
        self.y_mean = None
        self.y_std = None
        self.noise = noise

        # Instantiate means & kernels
        self.mean = MeanComposer(
            pipeline_space, *(means or []), fallback_default_mean=DEFAULT_MEAN
        )
        self.kernel = instantiate_kernel(pipeline_space, kernels, combine_kernel)

    def _build_input_tensor(self, x_configs: list[SearchSpace]):
        x_tensor = torch.zeros(
            (len(x_configs), self.tensor_size), dtype=torch.get_default_dtype()
        )
        for i_sample, sample in enumerate(x_configs):
            for hp_name, hp in sample.items():
                hp_shape = self.all_hp_shapes[hp_name]
                x_tensor[i_sample, hp_shape.begin : hp_shape.end] = hp.get_tensor_value(
                    hp_shape
                )
        return x_tensor

    def _build_output_tensor(self, y_values, set_y_scale=False):
        y_values = torch.tensor(y_values, dtype=torch.get_default_dtype())
        if set_y_scale:
            self.y_mean = y_values.mean()
            self.y_std = y_values.std()
            if self.y_std.abs() < EPSILON:
                self.y_std = EPSILON
        y_values = (y_values - self.y_mean) / self.y_std
        return y_values

    def fit(self, train_x, train_y):
        if not train_x:
            raise ValueError("Can't fit a GP on no data")

        # Compute the shape of the tensor and the bounds of each HP
        self.tensor_size = 0
        self.all_hp_shapes = {}
        for hp_name in train_x[0]:
            hp_instances = [sample[hp_name] for sample in train_x]
            hp_shape = hp_instances[0].get_tensor_shape(hp_instances)

            hp_shape.set_bounds(self.tensor_size)
            self.tensor_size = hp_shape.end
            self.all_hp_shapes[hp_name] = hp_shape

        # Build the input tensors

        x_tensor = self._build_input_tensor(train_x)
        y_tensor = self._build_output_tensor(train_y, set_y_scale=True)
        self.fitted_on = ((train_x, train_y), (x_tensor, y_tensor))

        # Then build the GPyTorch model
        gpytorch_kernel = self.kernel.build(self.all_hp_shapes)
        gpytorch_mean = self.mean.build(self.all_hp_shapes)
        self.gp = GPTorchModel(
            x_tensor, y_tensor, gpytorch_mean, gpytorch_kernel, noise=self.noise
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.gp.train()
        botorch.fit.fit_gpytorch_model(mll)
        self.gp.eval()

    def predict_distribution(self, x_configs, normalized=False):
        if self.gp is None:
            raise Exception("Can't use predict before fitting the GP model")

        x_tensor = self._build_input_tensor(x_configs)
        with torch.no_grad():
            with botorch.models.utils.gpt_posterior_settings():
                mvn = self.gp(x_tensor)
        mean, covariance_matrix = mvn.mean, mvn.covariance_matrix
        covariance_matrix = torch.minimum(covariance_matrix, torch.tensor(0))
        if not normalized:
            mean = mean * self.y_std + self.y_mean
            covariance_matrix = covariance_matrix * self.y_std**2
        return mean, covariance_matrix

    def predict(self, x_config: Iterable[SearchSpace] | SearchSpace, normalized=False):
        x = [x_config] if isinstance(x_config, SearchSpace) else x_config
        mean, cov = self.predict_distribution(x, normalized=normalized)

        if isinstance(x_config, SearchSpace):
            mean, cov = mean.reshape(tuple()), cov.reshape(tuple())
        return mean, cov

    def predict_mean(
        self, x_config: Iterable[SearchSpace] | SearchSpace, normalized=False
    ):
        """For quicker predictions of only the mean"""
        if isinstance(x_config, SearchSpace):
            return self.predict(x_config, normalized=normalized)[0]
        return torch.tensor(
            [self.predict(x, normalized=normalized)[0] for x in x_config],
            dtype=torch.get_default_dtype(),
        )
