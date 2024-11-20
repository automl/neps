import pytest
import torch
import networkx as nx
from gpytorch.kernels import MaternKernel, AdditiveKernel
from botorch.models.gp_regression_mixed import CategoricalKernel, ScaleKernel
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from grakel_replace.mixed_single_task_gp import MixedSingleTaskGP
from grakel_replace.torch_wl_kernel import TorchWLKernel


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n_samples = 5
    n_numerical = 2
    n_categorical = 2

    # Create numerical and categorical features
    X = torch.empty(size=(n_samples, n_numerical + n_categorical), dtype=torch.float64)
    X[:, :n_numerical] = torch.rand(size=(n_samples, n_numerical), dtype=torch.float64)
    X[:, n_numerical:] = torch.randint(0, 3, size=(n_samples, n_categorical),
                                       dtype=torch.float64)

    # Create sample graphs
    graphs = [nx.erdos_renyi_graph(n=4, p=0.5) for _ in range(n_samples)]

    # Create target values
    y = torch.rand(size=(n_samples, 1), dtype=torch.float64)

    return X, graphs, y


@pytest.fixture
def sample_kernels():
    """Create sample kernels for testing."""
    n_numerical = 2
    n_categorical = 2

    matern = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=n_numerical,
            active_dims=tuple(range(n_numerical)),
        ),
    )

    hamming = ScaleKernel(
        CategoricalKernel(
            ard_num_dims=n_categorical,
            active_dims=tuple(range(n_numerical, n_numerical + n_categorical)),
        ),
    )

    combined_kernel = AdditiveKernel(matern, hamming)
    wl_kernel = TorchWLKernel(n_iter=3, normalize=True)

    return combined_kernel, wl_kernel


def test_model_initialization_and_validation(sample_data, sample_kernels):
    """Test GP initialization, inputs validation, and basic properties."""
    X, graphs, y = sample_data
    combined_kernel, wl_kernel = sample_kernels

    # Test successful initialization with all parameters
    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=combined_kernel,
        wl_kernel=wl_kernel,
    )

    assert gp._train_graphs == graphs
    assert isinstance(gp._wl_kernel, TorchWLKernel)
    assert gp.num_cat_kernel == combined_kernel

    # Test empty input validation
    with pytest.raises(ValueError, match="Training inputs.*cannot be empty"):
        MixedSingleTaskGP(
            train_X=torch.empty((0, 4), dtype=torch.float64),
            train_graphs=[],
            train_Y=torch.empty((0, 1), dtype=torch.float64),
        )


def test_forward_pass_and_predictions(sample_data, sample_kernels):
    """Test forward pass, shape consistency, and prediction characteristics."""
    X, graphs, y = sample_data
    combined_kernel, wl_kernel = sample_kernels

    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=combined_kernel,
        wl_kernel=wl_kernel,
    )

    # Test forward pass with training data
    output = gp.forward(X, graphs)
    assert isinstance(output, MultivariateNormal)
    assert output.mean.shape == (len(X),)
    assert output.covariance_matrix.shape == (len(X), len(X))

    # Test forward pass with test data
    n_test = 3
    test_X = torch.rand(size=(n_test, X.shape[1]), dtype=torch.float64)
    test_graphs = [nx.erdos_renyi_graph(n=4, p=0.5) for _ in range(n_test)]

    output_test = gp.forward(test_X, test_graphs)
    assert output_test.mean.shape == (n_test,)
    assert output_test.covariance_matrix.shape == (n_test, n_test)

    # Test input validation
    # Mismatched number of features and graphs
    mismatched_test_X = torch.rand(size=(3, X.shape[1]), dtype=torch.float64)
    mismatched_test_graphs = [nx.erdos_renyi_graph(n=4, p=0.5) for _ in range(4)]

    with pytest.raises(ValueError,
                       match="Number of feature vectors.*must match.*number of graphs"):
        gp.forward(mismatched_test_X, mismatched_test_graphs)

    # Invalid graph input
    invalid_graphs = ["not_a_graph", 123, None, False, True]
    with pytest.raises(TypeError,
                       match="Expected input type is a list of NetworkX graphs."):
        gp.forward(X, invalid_graphs)


def test_kernel_combinations_and_properties(sample_data):
    """Test kernel combination, invariance, and consistency properties."""
    X, graphs, y = sample_data

    # Test kernel combination and variance changes
    # Create GP with only graph kernel
    gp_graph_only = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
    )

    output_graph = gp_graph_only.forward(X, graphs)
    graph_var = output_graph.variance

    # Create GP with combined kernels
    n_numerical = 2
    matern = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=n_numerical,
            active_dims=tuple(range(n_numerical)),
        ),
    )

    gp_combined = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=matern,
    )

    output_combined = gp_combined.forward(X, graphs)
    combined_var = output_combined.variance

    # Combined kernel should have larger variance due to addition
    assert torch.all(combined_var > graph_var)

    # Use graphs with slight variations to avoid singular matrix
    similar_graphs = [
        nx.complete_graph(5) for _ in range(len(graphs))
    ]

    # Add small random perturbations to make graphs slightly different
    for i in range(1, len(similar_graphs)):
        G = similar_graphs[i]
        # Add or remove edges with a small probability
        edges_to_add = [(u, v) for u in range(5) for v in range(u + 1, 5)
                        if not G.has_edge(u, v) and torch.rand(1) < 0.1]
        edges_to_remove = [(u, v) for (u, v) in G.edges()
                           if torch.rand(1) < 0.1]

        G.add_edges_from(edges_to_add)
        G.remove_edges_from(edges_to_remove)

    gp_similar = MixedSingleTaskGP(
        train_X=X,
        train_graphs=similar_graphs,
        train_Y=y,
    )

    # Compute kernel matrix and check diagonal consistency
    kernel_matrix = gp_similar._K_train
    diag = kernel_matrix.diag()

    # Allow for slight variations due to graph perturbations
    assert torch.allclose(diag, diag[0], atol=1e-1)

    # Check that the matrix is not completely uniform
    assert not torch.allclose(kernel_matrix, torch.ones_like(kernel_matrix), rtol=1e-5)


def test_model_prediction_consistency(sample_data, sample_kernels):
    """Test prediction consistency and mean prediction bounds."""
    X, graphs, y = sample_data
    combined_kernel, wl_kernel = sample_kernels

    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=combined_kernel,
        wl_kernel=wl_kernel,
    )

    # Multiple forward passes should give consistent results
    output1 = gp.forward(X, graphs)
    output2 = gp.forward(X, graphs)

    assert torch.allclose(output1.mean, output2.mean)
    assert torch.allclose(output1.variance, output2.variance)

    # Mean predictions should be within reasonable bounds
    with torch.no_grad():
        output = gp.forward(X, graphs)
        predictions = output.mean
        uncertainties = output.variance.sqrt()

        assert torch.all(predictions >= y.min() - 2 * uncertainties)
        assert torch.all(predictions <= y.max() + 2 * uncertainties)


def test_graph_kernel_caching(sample_data, sample_kernels):
    """Test that graph kernel matrices are properly cached."""
    X, graphs, y = sample_data
    combined_kernel, wl_kernel = sample_kernels

    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=combined_kernel,
        wl_kernel=wl_kernel,
    )

    # First forward pass
    _ = gp.forward(X, graphs)
    K_train_1 = gp._K_train.clone()

    # Second forward pass
    _ = gp.forward(X, graphs)
    K_train_2 = gp._K_train.clone()

    # Cached kernel matrices should be identical
    assert torch.allclose(K_train_1, K_train_2)


def test_large_dataset_handling():
    """Test the model's behavior with large datasets."""
    n_samples = 100
    n_features = 4

    # Create large numerical and categorical features
    X = torch.rand(size=(n_samples, n_features), dtype=torch.float64)

    # Create a large set of random graphs
    graphs = [nx.erdos_renyi_graph(n=10, p=0.2) for _ in range(n_samples)]

    # Create target values
    y = torch.rand(size=(n_samples, 1), dtype=torch.float64)

    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
    )

    output = gp.forward(X, graphs)
    assert output.mean.shape == (n_samples,)
    assert output.covariance_matrix.shape == (n_samples, n_samples)
