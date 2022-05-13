from __future__ import annotations

from typing import Iterable

import botorch
import gpytorch
import torch

from ....search_spaces.search_space import SearchSpace
from ..default_consts import DEFAULT_MEAN, EPSILON
from ..kernels import Kernel, instantiate_kernel
from ..means import GpMean, MeanComposer


class GPTorchModel(gpytorch.models.ExactGP):
    MIN_INFERRED_NOISE_LEVEL = 1e-4

    def __init__(self, train_x, train_y, mean, kernel):
        noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=gpytorch.constraints.GreaterThan(
                self.MIN_INFERRED_NOISE_LEVEL,
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
        combine_kernel: str = "product",
        logger=None,
    ):
        self.logger = logger
        self.gp = None
        self.fitted_on = None
        self.tensor_size = None
        self.all_hp_shapes = None
        self.y_mean = None
        self.y_std = None

        # Instantiate means & kernels
        self.mean = MeanComposer(
            pipeline_space, *(means or []), fallback_default_mean=DEFAULT_MEAN
        )
        self.kernel = instantiate_kernel(pipeline_space, kernels, combine_kernel)

    def _build_input_tensor(self, x_configs):
        assert isinstance(x_configs, list)
        x_tensor = torch.zeros((len(x_configs), self.tensor_size))
        for i_sample, sample in enumerate(x_configs):
            for hp_name, hp in sample.items():
                hp_shape = self.all_hp_shapes[hp_name]
                x_tensor[i_sample, hp_shape.begin : hp_shape.end] = hp.get_tensor_value(
                    hp_shape
                )
        return x_tensor

    def _build_output_tensor(self, y_values, set_y_scale=False):
        y_values = torch.tensor(y_values)
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
        self.gp = GPTorchModel(x_tensor, y_tensor, gpytorch_mean, gpytorch_kernel)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.gp.train()
        botorch.fit.fit_gpytorch_model(mll)
        self.gp.eval()

    def predict_distribution(self, x_configs):
        if self.gp is None:
            raise Exception("Can't use predict before fitting the GP model")

        x_tensor = self._build_input_tensor(x_configs)
        with torch.no_grad():
            mvn = self.gp(x_tensor)
        mvn = gpytorch.distributions.MultivariateNormal(
            mvn.mean * self.y_std + self.y_mean, mvn.covariance_matrix * self.y_std**2
        )
        return mvn

    def predict(self, x_config: Iterable[SearchSpace] | SearchSpace):
        x = [x_config] if isinstance(x_config, SearchSpace) else x_config
        mvn = self.predict_distribution(x)
        mean, cov = mvn.mean, mvn.covariance_matrix

        if isinstance(x_config, SearchSpace):
            mean, cov = mean.reshape(tuple()), cov.reshape(tuple())
        return mean, cov

    def predict_mean(self, x_config: Iterable[SearchSpace] | SearchSpace):
        """For quicker predictions of only the mean"""
        if isinstance(x_config, SearchSpace):
            return self.predict(x_config)[0]
        return torch.tensor([self.predict(x)[0] for x in x_config])
