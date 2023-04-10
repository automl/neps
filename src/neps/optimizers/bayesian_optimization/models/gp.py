# type: ignore
from __future__ import annotations

from functools import partial
from typing import Iterable

import botorch
import gpytorch
import torch
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.kernels import ScaleKernel
from typing_extensions import Literal

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
from ..utils import GpAuxData

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
        hierarchy_consider=None,
        d_graph_features=1,
        vectorial_features=None,
        verbose=False,
    ):  # pylint: disable=unused-argument
        self.logger = logger
        self.gp = None
        self.fitted_on = None
        self.tensor_size = None
        self.all_hp_shapes = None
        self.y_mean = None
        self.y_std = None
        self.noise = noise
        self.graph_structures = []
        self.hp_hierarchy_levels = []

        self.train_size = None

        self.d_graph_features = d_graph_features

        self.aux_data = GpAuxData(pipeline_space, hierarchy_consider, d_graph_features)

        # Instantiate means & kernels
        self.mean = MeanComposer(
            pipeline_space, *(means or []), fallback_default_mean=DEFAULT_MEAN
        )
        # Extend the pipeline_space when hierarchical kernels are provided
        if kernels is not None and len(hierarchy_consider or []) + 1 <= len(kernels):
            self.aux_data.extend_hierarchical_space()
            pipeline_space = self.aux_data.extended_pipeline_space

        self.kernel = instantiate_kernel(pipeline_space, kernels, combine_kernel)

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
        # Use to throw an error if train_size == test_size see commit 90aebb5 for details
        self.train_size = len(train_x)

        # Compute the shape of the input tensor
        self.aux_data.reset()
        for hp_idx, hp_name in enumerate(train_x[0]):
            self.aux_data.add_hp(train_x, hp_idx, hp_name)

        x_tensor, x_graphs = self.aux_data.build_input_tensor(train_x)

        self.aux_data.insert_graph_data(x_graphs)
        y_tensor = self._build_output_tensor(train_y, set_y_scale=True)
        self.fitted_on = ((train_x, train_y), (x_tensor, x_graphs, y_tensor))

        if self.aux_data.hierarchy_consider:
            self.kernel.assign_hierarichal_hyperparameters(self.aux_data.hierarchical_hps)

        if self.aux_data.d_graph_features > 0:
            self.kernel.assign_feature_hyperparameters(
                self.aux_data.d_graph_feature_hp_names
            )

        # Then build the GPyTorch model
        # dirty trick to inject the graph data into the kernel...
        gpytorch_kernel = self.kernel.build(self.aux_data.all_hp_shapes)
        gpytorch_mean = self.mean.build(self.aux_data.all_hp_shapes)
        self.gp = GPTorchModel(
            x_tensor, y_tensor, gpytorch_mean, gpytorch_kernel, noise=self.noise
        )

        # Init torch optimizer and optimizer options
        optimizer_fn = partial(
            fit_gpytorch_torch, options={"maxiter": 20, "disp": False, "lr": 0.05}
        )

        if self.aux_data.graph_structures is not None:  # pre-compute graph kernel
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
        botorch.fit.fit_gpytorch_model(mll, optimizer=optimizer_fn)
        self.gp.eval()

    def predict_distribution(self, x_configs, normalized=False):
        if self.gp is None:
            raise Exception("Can't use predict before fitting the GP model")
        x_tensor, x_graphs = self.aux_data.build_input_tensor(x_configs)
        # gpytorch.settings.max_eager_kernel_size._set_value(1000)

        # inject x_graphs into kernel
        if self.aux_data.graph_structures is not None:  # pre-compute graph kernel
            graph_kernel_idx = 0
            for kernel in self.gp.covar_module.kernels:
                if isinstance(kernel, GraphKernel) or (
                    isinstance(kernel, ScaleKernel)
                    and isinstance(kernel.base_kernel, GenericGPyTorchStationaryKernel)
                    and isinstance(kernel.base_kernel.neps_kernel, GraphKernel)
                ):
                    if not self.aux_data.hierarchy_consider:
                        kernel.base_kernel.neps_kernel.set_eval_graphs(x_graphs)
                    elif graph_kernel_idx < len(self.aux_data.graph_structures):
                        kernel.base_kernel.neps_kernel.set_eval_graphs(
                            [x_graphs[graph_kernel_idx]]
                        )
                        graph_kernel_idx += 1
                    else:
                        Exception(
                            f"Graph kernels ({len(self.gp.covar_module.kernels)}) can't be more "
                            f"than graph structures ({len(self.aux_data.graph_structures)})"
                        )
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

        assert len(x_config) != self.train_size, (
            "Can't have train and test batch "
            "sizes equal, see commit "
            "[gpytorch 90aebb5] for details"
        )
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
