from __future__ import annotations

from itertools import product

import networkx as nx
import pytest
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import LinearMCObjective, qLogNoisyExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.kernels import CategoricalKernel
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import AdditiveKernel, MaternKernel, ScaleKernel

from neps.optimizers.models.graphs.context_managers import set_graph_lookup
from neps.optimizers.models.graphs.kernels import BoTorchWLKernel
from neps.optimizers.models.graphs.optimization import optimize_acqf_graph, sample_graphs
from neps.optimizers.models.graphs.utils import min_max_scale


class TestGraphOptimizationPipeline:
    @pytest.fixture
    def setup_data(self) -> dict:
        """Fixture to set up common data for tests."""
        TRAIN_CONFIGS = 50
        TEST_CONFIGS = 10
        TOTAL_CONFIGS = TRAIN_CONFIGS + TEST_CONFIGS

        N_NUMERICAL = 2
        N_CATEGORICAL = 1
        N_CATEGORICAL_VALUES_PER_CATEGORY = 2
        N_GRAPH = 1

        # Generate random data
        X = torch.cat([
            torch.rand((TOTAL_CONFIGS, N_NUMERICAL), dtype=torch.float64),
            torch.randint(0, N_CATEGORICAL_VALUES_PER_CATEGORY,
                          (TOTAL_CONFIGS, N_CATEGORICAL), dtype=torch.float64),
            torch.arange(TOTAL_CONFIGS, dtype=torch.float64).unsqueeze(1)
        ], dim=1)

        # Generate random graphs
        graphs = [nx.erdos_renyi_graph(5, 0.5) for _ in range(TOTAL_CONFIGS)]

        # Generate random target values
        y = torch.rand(TOTAL_CONFIGS, dtype=torch.float64) + 0.5

        # Split into train and test sets
        train_x, test_x = X[:TRAIN_CONFIGS], X[TRAIN_CONFIGS:]
        train_graphs, test_graphs = graphs[:TRAIN_CONFIGS], graphs[TRAIN_CONFIGS:]
        train_y, test_y = y[:TRAIN_CONFIGS].unsqueeze(-1), y[TRAIN_CONFIGS:].unsqueeze(-1)

        # Scale the data
        train_x, test_x = min_max_scale(train_x), min_max_scale(test_x)

        return {
            "train_x": train_x,
            "test_x": test_x,
            "train_graphs": train_graphs,
            "test_graphs": test_graphs,
            "train_y": train_y,
            "test_y": test_y,
            "N_NUMERICAL": N_NUMERICAL,
            "N_CATEGORICAL": N_CATEGORICAL,
            "N_CATEGORICAL_VALUES_PER_CATEGORY": N_CATEGORICAL_VALUES_PER_CATEGORY,
            "N_GRAPH": N_GRAPH,
        }

    def test_gp_fit_and_predict(self, setup_data: dict) -> None:
        """Test fitting the GP and making predictions."""
        train_x = setup_data["train_x"]
        train_y = setup_data["train_y"]
        test_x = setup_data["test_x"]
        train_graphs = setup_data["train_graphs"]
        setup_data["test_graphs"]

        # Define the kernels
        kernels = [
            ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=setup_data["N_NUMERICAL"],
                                     active_dims=range(setup_data["N_NUMERICAL"]))),
            ScaleKernel(
                CategoricalKernel(ard_num_dims=setup_data["N_CATEGORICAL"],
                                  active_dims=range(setup_data["N_NUMERICAL"],
                                                    setup_data["N_NUMERICAL"] +
                                                    setup_data["N_CATEGORICAL"])
                                  )
            ),
            ScaleKernel(
                BoTorchWLKernel(graph_lookup=train_graphs, n_iter=5, normalize=True,
                                active_dims=(train_x.shape[1] - 1,)))
        ]

        # Create the GP model
        gp = SingleTaskGP(train_X=train_x, train_Y=train_y,
                          covar_module=AdditiveKernel(*kernels))

        # Fit the GP
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Make predictions on the test set
        with torch.no_grad():
            posterior = gp.forward(test_x)
            predictions = posterior.mean
            uncertainties = posterior.variance.sqrt()

            # Ensure predictions are in the correct shape (10, 1)
            predictions = predictions.unsqueeze(-1)  # Reshape to (10, 1)

        # Basic checks
        assert predictions.shape == (setup_data["test_x"].shape[0], 1)
        assert uncertainties.shape == (setup_data["test_x"].shape[0],)

    def test_acquisition_function_optimization(self, setup_data: dict) -> None:
        """Test optimizing the acquisition function with graph sampling."""
        train_x = setup_data["train_x"]
        train_y = setup_data["train_y"]
        train_graphs = setup_data["train_graphs"]

        # Define the kernels
        kernels = [
            ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=setup_data["N_NUMERICAL"],
                                     active_dims=range(setup_data["N_NUMERICAL"]))),
            ScaleKernel(
                CategoricalKernel(
                    ard_num_dims=setup_data["N_CATEGORICAL"],
                    active_dims=range(setup_data["N_NUMERICAL"],
                                      setup_data["N_NUMERICAL"] +
                                      setup_data["N_CATEGORICAL"])
                )
            ),
            ScaleKernel(
                BoTorchWLKernel(graph_lookup=train_graphs, n_iter=5, normalize=True,
                                active_dims=(train_x.shape[1] - 1,)))
        ]

        # Create the GP model
        gp = SingleTaskGP(train_X=train_x, train_Y=train_y,
                          covar_module=AdditiveKernel(*kernels))

        # Fit the GP
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Define the acquisition function
        acq_function = qLogNoisyExpectedImprovement(
            model=gp,
            X_baseline=train_x,
            objective=LinearMCObjective(weights=torch.tensor([-1.0])),
            prune_baseline=True,
        )

        # Define bounds for optimization
        bounds = torch.tensor([
            [0.0] * setup_data["N_NUMERICAL"] + [0.0] * setup_data["N_CATEGORICAL"] + [
                -1.0] * setup_data["N_GRAPH"],
            [1.0] * setup_data["N_NUMERICAL"] + [
                float(setup_data["N_CATEGORICAL_VALUES_PER_CATEGORY"] - 1)] * setup_data[
                "N_CATEGORICAL"] + [len(train_x) - 1] * setup_data["N_GRAPH"],
        ])

        # Define fixed categorical features
        cats_per_column = {i: list(range(setup_data["N_CATEGORICAL_VALUES_PER_CATEGORY"]))
                           for i in range(setup_data["N_NUMERICAL"],
                                          setup_data["N_NUMERICAL"] + setup_data[
                                              "N_CATEGORICAL"])}
        fixed_cats = [dict(zip(cats_per_column.keys(), combo, strict=False)) for combo in
                      product(*cats_per_column.values())]

        # Optimize the acquisition function
        best_candidate, best_graph, best_score = optimize_acqf_graph(
            acq_function=acq_function,
            bounds=bounds,
            fixed_features_list=fixed_cats,
            train_graphs=train_graphs,
            num_graph_samples=2,
            num_restarts=2,
            raw_samples=16,
            q=1,
        )

        # Assertions for the acquisition function optimization
        assert isinstance(best_candidate,
                          torch.Tensor), "Best candidate should be a tensor"
        assert best_candidate.shape == (1, train_x.shape[1] - 1), \
            "Best candidate should have the correct shape (excluding the graph index)"
        assert isinstance(best_graph, nx.Graph), "Best graph should be a NetworkX graph"
        assert isinstance(best_score, float), "Best score should be a float"

        # Ensure the best candidate does not contain the graph index column
        assert best_candidate.shape[1] == train_x.shape[1] - 1, \
            "Best candidate should not include the graph index column"

    def test_graph_sampling(self, setup_data: dict) -> None:
        """Test the graph sampling functionality."""
        train_graphs = setup_data["train_graphs"]
        num_samples = 5

        # Sample graphs
        sampled_graphs = sample_graphs(train_graphs, num_samples=num_samples)

        # Basic checks
        assert len(sampled_graphs) == num_samples, \
            f"Expected {num_samples} sampled graphs, got {len(sampled_graphs)}"
        assert all(isinstance(graph, nx.Graph) for graph in sampled_graphs), \
            "All sampled graphs should be NetworkX graphs"
        assert all(nx.is_connected(graph) for graph in sampled_graphs), \
            "All sampled graphs should be connected"

    def test_min_max_scaling(self, setup_data: dict) -> None:
        """Test the min-max scaling utility."""
        train_x = setup_data["train_x"]

        # Apply min-max scaling
        scaled_train_x = min_max_scale(train_x)

        # Assertions for min-max scaling
        assert torch.all(scaled_train_x >= 0), "Scaled values should be >= 0"
        assert torch.all(scaled_train_x <= 1), "Scaled values should be <= 1"
        assert scaled_train_x.shape == train_x.shape, \
            "Scaled data should have the same shape as the input data"

        # Check that the scaling is correct
        for i in range(train_x.shape[1]):
            col_min = torch.min(train_x[:, i])
            col_max = torch.max(train_x[:, i])
            if col_min != col_max:  # Avoid division by zero
                expected_scaled_col = (train_x[:, i] - col_min) / (col_max - col_min)
                assert torch.allclose(scaled_train_x[:, i], expected_scaled_col), \
                    f"Scaling is incorrect for column {i}"

    def test_set_graph_lookup(self, setup_data: dict) -> None:
        """Test the set_graph_lookup context manager."""
        train_graphs = setup_data["train_graphs"]
        test_graphs = setup_data["test_graphs"]

        # Define the kernel
        kernel = BoTorchWLKernel(graph_lookup=train_graphs, n_iter=5, normalize=True,
                                 active_dims=(0,))

        # Use the context manager to temporarily set the graph lookup
        with set_graph_lookup(kernel, test_graphs, append=True):
            assert len(kernel.graph_lookup) == len(train_graphs) + len(test_graphs)

        # Check that the original graph lookup is restored
        assert len(kernel.graph_lookup) == len(train_graphs)
