from __future__ import annotations

from typing import Iterable

import botorch
import gpytorch
import torch
from gpytorch.kernels import ScaleKernel
from typing_extensions import Literal

from ....search_spaces.graph_grammar.graph import Graph
from ....search_spaces.parameter import HpTensorShape
from ....search_spaces.search_space import SearchSpace
from ..default_consts import (
    DEFAULT_COMBINE,
    DEFAULT_MEAN,
    EPSILON,
    MIN_INFERRED_NOISE_LEVEL,
)
from ..kernels import Kernel, instantiate_kernel
from ..kernels.base_kernel import GenericGPyTorchStationaryKernel
from ..kernels.graph_kernel import GraphKernel
from ..means import GPMean, MeanComposer

DUMMY_VAL = -1


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
        means: Iterable[GPMean] = None,
        kernels: Iterable[Kernel] = None,
        combine_kernel: str = DEFAULT_COMBINE,
        logger=None,
        noise: str = "low",
    ):
        self.logger = logger
        self.gp = None
        self.fitted_on = None
        self.tensor_size = None
        self.all_hp_shapes = None
        self.y_mean = None
        self.y_std = None
        self.noise = noise
        self.graph_structures = []

        # Instantiate means & kernels
        self.mean = MeanComposer(
            pipeline_space, *(means or []), fallback_default_mean=DEFAULT_MEAN
        )
        self.kernel = instantiate_kernel(pipeline_space, kernels, combine_kernel)

    def _build_input_tensor(
        self, x_configs: list[SearchSpace]
    ) -> list[torch.tensor, list[Graph]]:
        x_tensor = (
            torch.ones(
                (len(x_configs), self.tensor_size), dtype=torch.get_default_dtype()
            )
            * DUMMY_VAL
        )
        if self.graph_structures is not None:
            x_graphs = [[] for _ in range(len(self.graph_structures))]
        else:
            x_graphs = None
        for i_sample, sample in enumerate(x_configs):
            graph_structure_idx = 0
            for hp_idx, (hp_name, hp) in enumerate(sample.items()):
                hp_shape = self.all_hp_shapes[hp_name]
                if hp_idx not in self.graph_structures:
                    x_tensor[
                        i_sample, hp_shape.begin : hp_shape.end
                    ] = hp.get_tensor_value(hp_shape)
                else:
                    x_graphs[graph_structure_idx].append(hp.get_tensor_value())
                    graph_structure_idx += 1
        return x_tensor, x_graphs

    def _build_output_tensor(
        self, y_values: list[float], set_y_scale: bool = False
    ) -> torch.tensor:
        y_values = torch.tensor(y_values, dtype=torch.get_default_dtype())
        if set_y_scale:
            self.y_mean = y_values.mean()
            self.y_std = y_values.std()
            if self.y_std.abs() < EPSILON:
                self.y_std = EPSILON
        y_values = (y_values - self.y_mean) / self.y_std
        return y_values

    def fit(self, train_x: list, train_y: list):
        if not train_x or not train_y:
            raise ValueError("Can't fit a GP on no data")
        if len(train_x) != len(train_y):
            raise ValueError("Can't fit a GP on data with different x and y values")

        # Compute the shape of the tensor and the bounds of each HP
        self.tensor_size = 0
        self.all_hp_shapes = {}
        self.graph_structures = []
        for hp_idx, hp_name in enumerate(train_x[0]):
            hp_instances = [sample[hp_name] for sample in train_x]
            hp_shape = hp_instances[0].get_tensor_shape(hp_instances)
            if hp_shape is None:
                self.graph_structures.append(hp_idx)
                hp_shape = HpTensorShape(length=1, hp_instances=hp_instances)
            hp_shape.set_bounds(self.tensor_size)
            self.tensor_size = hp_shape.end
            self.all_hp_shapes[hp_name] = hp_shape

        # Build the input tensors
        x_tensor, x_graphs = self._build_input_tensor(train_x)
        if self.graph_structures is not None:
            for hp_shape in self.all_hp_shapes.values():
                for active_dim in hp_shape.active_dims:
                    if active_dim in self.graph_structures:
                        hp_shape.hp_instances = x_graphs[active_dim]
        y_tensor = self._build_output_tensor(train_y, set_y_scale=True)
        self.fitted_on = ((train_x, train_y), (x_tensor, x_graphs, y_tensor))

        # Then build the GPyTorch model
        # dirty trick to inject the graph data into the kernel...
        gpytorch_kernel = self.kernel.build(self.all_hp_shapes)
        gpytorch_mean = self.mean.build(self.all_hp_shapes)
        self.gp = GPTorchModel(
            x_tensor, y_tensor, gpytorch_mean, gpytorch_kernel, noise=self.noise
        )

        if self.graph_structures is not None:  # pre-compute graph kernel
            for kernel in self.gp.covar_module.kernels:
                if isinstance(kernel, GraphKernel) or (
                    isinstance(kernel, ScaleKernel)
                    and isinstance(kernel.base_kernel, GenericGPyTorchStationaryKernel)
                    and isinstance(kernel.base_kernel.neps_kernel, GraphKernel)
                ):
                    kernel.base_kernel.neps_kernel.prefit_graph_kernel(
                        y=y_tensor, likelihood=self.gp.likelihood.noise.item()
                    )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.gp.train()
        botorch.fit.fit_gpytorch_model(mll)
        self.gp.eval()

    def predict_distribution(self, x_configs, normalized=False):
        if self.gp is None:
            raise Exception("Can't use predict before fitting the GP model")

        x_tensor, x_graphs = self._build_input_tensor(x_configs)
        # inject x_graphs into kernel
        if self.graph_structures is not None:  # pre-compute graph kernel
            for kernel in self.gp.covar_module.kernels:
                if isinstance(kernel, GraphKernel) or (
                    isinstance(kernel, ScaleKernel)
                    and isinstance(kernel.base_kernel, GenericGPyTorchStationaryKernel)
                    and isinstance(kernel.base_kernel.neps_kernel, GraphKernel)
                ):
                    kernel.base_kernel.neps_kernel.set_eval_graphs(x_graphs)
        with torch.no_grad():
            with botorch.models.utils.gpt_posterior_settings():
                mvn = self.gp(x_tensor)

                mean, covariance_matrix = mvn.mean, mvn.covariance_matrix
                covariance_matrix = torch.maximum(covariance_matrix, torch.tensor(0))
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
